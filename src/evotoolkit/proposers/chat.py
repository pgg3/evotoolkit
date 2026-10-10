"""Chat completion adapter and the three artifact output formats."""

import re
from pathlib import Path

from evotoolkit.core.types import Proposal, ProposalContext
from evotoolkit.evaluators.files import file_path

_FENCE = re.compile(r"^```([^\r\n]*)\r?\n(.*?)^```[ \t]*\r?$", re.MULTILINE | re.DOTALL)


def context_text(ctx: ProposalContext, instruction=None) -> str:
    instruction = instruction or ctx.instruction
    sections = [ctx.spec.prompt, instruction.text.format(modality=ctx.spec.modality, main_file=ctx.spec.main_file)]
    for label, solution in [(f"Parent {i + 1}", parent) for i, parent in enumerate(ctx.parents)] + ([("Current best", ctx.best)] if ctx.best else []):
        sections.append(f"{label}: {solution.name}\nScore: {solution.score}\nRationale: {solution.rationale}")
        sections.extend(f"```{path}\n{content}\n```" for path, content in solution.files.items())
        if solution.evaluation and solution.evaluation.feedback:
            sections.append(f"Feedback: {solution.evaluation.feedback}")
    if ctx.hints:
        sections.append("Insights:\n" + "\n".join(ctx.hints))
    return "\n\n".join(sections)


def _parse(text: str, output_format: str, main_file: str):
    blocks = list(_FENCE.finditer(text))
    if not blocks:
        raise ValueError("Response contains no fenced artifact")
    name = rationale = ""
    if output_format == "files":
        files = {}
        paths = set()
        for block in blocks:
            path = block.group(1).strip()
            target = file_path(Path.cwd(), path)
            if target in paths:
                raise ValueError(f"Duplicate artifact path: {path}")
            paths.add(target)
            files[path] = block.group(2).removesuffix("\n").removesuffix("\r")
    else:
        if len(blocks) != 1:
            raise ValueError("Single-file output must contain exactly one fenced block")
        file_path(Path.cwd(), main_file)
        files = {main_file: blocks[0].group(2).removesuffix("\n").removesuffix("\r")}
        before, after = text[: blocks[0].start()], text[blocks[0].end() :]
        if output_format == "name_code_thought":
            name_match = re.search(r"^name:[ \t]*(.+)$", before, re.MULTILINE | re.IGNORECASE)
            thought_match = re.search(r"^thought:[ \t]*(.*)", after, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if not name_match or not thought_match:
                raise ValueError("Expected name: before code and thought: after code")
            name, rationale = name_match.group(1).strip(), thought_match.group(1).strip()
        else:
            thought_match = re.search(r"\{(.*?)\}", before, re.DOTALL)
            rationale = thought_match.group(1).strip() if thought_match else ""
    return files, name, rationale


class ChatProposer:
    def __init__(self, llm, templates=None, *, eventlog=None):
        self.llm = llm
        self.templates = templates or {}
        self.eventlog = eventlog
        self.name = f"chat:{getattr(llm, '_model', type(llm).__name__)}"

    def propose(self, ctx: ProposalContext) -> list[Proposal]:
        instruction = self.templates.get(ctx.instruction.name, ctx.instruction)
        formats = {
            "code_block": f"Return the complete content of {ctx.spec.main_file} in one fenced block.",
            "name_code_thought": f"Return name: descriptive_name, then code: followed by one fenced block containing {ctx.spec.main_file}, then thought: your rationale.",
            "files": "Return each complete file in a separate fenced block, with its relative path immediately after the opening ``` (for example ```src/main.py).",
        }
        if instruction.output_format not in formats:
            raise ValueError(f"Unknown output format: {instruction.output_format}")
        if ctx.num_proposals < 1:
            raise ValueError("num_proposals must be positive")
        prompt = context_text(ctx, instruction) + "\n\n" + formats[instruction.output_format]
        if ctx.num_proposals > 1:
            prompt += f"\nReturn exactly {ctx.num_proposals} alternatives, separated by a line containing only ===PROPOSAL===."
        messages = [{"role": "user", "content": prompt}]
        call_id = ctx.call_id
        trace = {"messages": messages}
        try:
            response, usage = self.llm.get_response(messages)
            trace.update(response=response, usage=usage)
            responses = re.split(r"^===PROPOSAL===[ \t]*\r?$", response, flags=re.MULTILINE)
            if len(responses) != ctx.num_proposals:
                raise ValueError(f"Expected {ctx.num_proposals} proposals, received {len(responses)}")
            proposals = []
            for response in responses:
                files, name, rationale = _parse(response, instruction.output_format, ctx.spec.main_file)
                proposals.append(Proposal(call_id, files, name, rationale, dict(usage)))
            return proposals
        except Exception as exc:
            trace["error"] = str(exc)
            raise
        finally:
            if self.eventlog:
                self.eventlog.trace(call_id, trace)
