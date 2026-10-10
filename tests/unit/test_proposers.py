import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evotoolkit.core import EvaluationResult, Runner
from evotoolkit.core.eventlog import EventLog
from evotoolkit.core.types import Instruction, ProposalContext, TaskSpec
from evotoolkit.proposers import INSTRUCTIONS, AgentProposer, ChatProposer
from evotoolkit.strategies import EoHStrategy


def _ctx(output_format="code_block", num_proposals=1):
    return ProposalContext(
        TaskSpec("test", "Produce a solution.", "python", "solution.py"),
        [],
        Instruction("test", 0, "Create {modality} in {main_file}", output_format),
        num_proposals=num_proposals,
        call_id="ctx-call",
    )


class ScriptedLLM:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def get_response(self, messages):
        self.messages = messages
        return self.response, {"total_tokens": 17}


def test_chat_multiple_proposals_share_usage_and_trace(tmp_path):
    llm = ScriptedLLM("{linear}\n```python\nx = 1\n```\n===PROPOSAL===\n```python\nx = 2\n```")
    with EventLog(tmp_path) as log:
        proposals = ChatProposer(llm, eventlog=log).propose(_ctx(num_proposals=2))
    assert [p.files for p in proposals] == [{"solution.py": "x = 1"}, {"solution.py": "x = 2"}]
    assert proposals[0].call_id == proposals[1].call_id
    assert proposals[0].rationale == "linear"
    assert proposals[0].usage == {"total_tokens": 17}
    trace = json.loads((tmp_path / "traces" / f"{proposals[0].call_id}.json").read_text())
    assert trace["response"] == llm.response


def test_chat_parses_name_code_thought_without_matching_code_labels():
    response = 'name: candidate\ncode:\n```python\ns = "thought: part of code"\n```\nthought: Improve the score.'
    proposal = ChatProposer(ScriptedLLM(response)).propose(_ctx("name_code_thought"))[0]
    assert proposal.name == "candidate"
    assert proposal.rationale == "Improve the score."
    assert "thought: part of code" in proposal.files["solution.py"]


def test_chat_multifile_preserves_whitespace():
    proposal = ChatProposer(ScriptedLLM("```src/main.py\n  value = 1  \n```\n```config.json\n{}\n```")).propose(_ctx("files"))[0]
    assert proposal.files == {"src/main.py": "  value = 1  ", "config.json": "{}"}


def test_chat_accepts_crlf_fences():
    proposal = ChatProposer(ScriptedLLM("```python\r\nx = 1\r\n```\r\n")).propose(_ctx())[0]
    assert proposal.files == {"solution.py": "x = 1"}


@pytest.mark.parametrize(
    "response,fmt",
    [("no code", "code_block"), ("```python\nx\n```", "name_code_thought"), ("```../x\ny\n```", "files"), ("```x\n1\n```\n```x\n2\n```", "files")],
)
def test_chat_rejects_malformed_output_and_keeps_trace(tmp_path, response, fmt):
    with EventLog(tmp_path) as log, pytest.raises(ValueError):
        ChatProposer(ScriptedLLM(response), eventlog=log).propose(_ctx(fmt))
    assert (tmp_path / "traces" / "ctx-call.json").exists()


def test_chat_template_override_and_string_modality():
    ctx = _ctx()
    ctx.spec = replace(ctx.spec, modality="string", main_file="prompt.txt")
    llm = ScriptedLLM("```text\nSay hello.\n```")
    templates = {"test": Instruction("test", 0, "Override for {modality}", "code_block")}
    proposal = ChatProposer(llm, templates).propose(ctx)[0]
    assert proposal.files == {"prompt.txt": "Say hello."}
    assert "Override for string" in llm.messages[0]["content"]


def test_builtin_templates_are_complete_and_format():
    assert set(INSTRUCTIONS) == {"i1", "e1", "e2", "m1", "m2", "funsearch", "init", "crossover", "mutation"}
    for instruction in INSTRUCTIONS.values():
        assert instruction.text.format(modality="python", main_file="solution.py")


def test_agent_headless_contract_workspace_usage_and_trace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "evaluate.py").write_text("print('evaluate')")
    (workspace / "dataset.bin").write_bytes(b"\xff\x00")

    @contextmanager
    def factory(ctx):
        yield workspace

    monkeypatch.setattr("evotoolkit.proposers.agent.shutil.which", lambda name: "headless")

    def run(command, **kwargs):
        assert command[:2] == ["headless", "codex"]
        assert command[2:6] == ["--prompt-file", str(workspace / "TASK.md"), "--work-dir", str(workspace)]
        assert command[6:8] == ["--json", "--usage"]
        assert "python evaluate.py" in (workspace / "TASK.md").read_text()
        (workspace / "solution.py").write_text("x = 42")
        (workspace / "agent.log").write_text("sensitive tool output")
        (workspace / "credentials.json").write_text('{"key":"secret"}')
        return subprocess.CompletedProcess(command, 0, '{"type":"turn.completed"}\n{"usage":{"usageStatus":"reported","totalTokens":23}}\n', "")

    monkeypatch.setattr("evotoolkit.proposers.agent.subprocess.run", run)
    with EventLog(tmp_path / "log") as log:
        proposal = AgentProposer("codex", factory, eventlog=log).propose(_ctx())[0]
    assert proposal.files == {"solution.py": "x = 42"}
    assert proposal.usage["totalTokens"] == 23
    assert list((tmp_path / "log" / "traces").glob("*.json"))


def test_agent_timeout_is_traced(tmp_path, monkeypatch):
    monkeypatch.setattr("evotoolkit.proposers.agent.shutil.which", lambda name: "headless")

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1, output=b"partial")

    monkeypatch.setattr("evotoolkit.proposers.agent.subprocess.run", run)
    with EventLog(tmp_path) as log, pytest.raises(subprocess.TimeoutExpired):
        AgentProposer("codex", timeout=1, eventlog=log).propose(_ctx())
    trace = json.loads((tmp_path / "traces" / "ctx-call.json").read_text())
    assert trace["stdout"] == "partial"


def test_agent_optional_evaluation_wrapper(monkeypatch):
    monkeypatch.setattr("evotoolkit.proposers.agent.shutil.which", lambda name: "headless")

    def run(command, **kwargs):
        root = Path(kwargs["cwd"])
        assert "subprocess.call(['python', 'solution.py'])" in (root / "evaluate.py").read_text()
        (root / "solution.py").write_text("print(42)")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("evotoolkit.proposers.agent.subprocess.run", run)
    ctx = _ctx()
    ctx.spec.extras["evaluation_command"] = ["python", "solution.py"]
    assert AgentProposer("codex").propose(ctx)[0].files == {"solution.py": "print(42)"}


@pytest.mark.parametrize(
    "names",
    [
        ["solution.py", "../secret"],
        ["solution.py", "/secret"],
        ["solution.py", "C:/secret"],
        ["solution.py", "solution.py"],
        ["solution.py", "evaluate.py"],
        ["solution.py", "ARTIFACTS.json"],
        {"files": ["solution.py"]},
    ],
)
def test_agent_rejects_unsafe_manifests(names, monkeypatch):
    monkeypatch.setattr("evotoolkit.proposers.agent.shutil.which", lambda name: "headless")

    def run(command, **kwargs):
        root = Path(kwargs["cwd"])
        (root / "solution.py").write_text("x = 42")
        (root / "ARTIFACTS.json").write_text(json.dumps(names))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("evotoolkit.proposers.agent.subprocess.run", run)
    with pytest.raises(ValueError):
        AgentProposer("codex").propose(_ctx())


def test_agent_manifest_collects_only_declared_files(monkeypatch):
    monkeypatch.setattr("evotoolkit.proposers.agent.shutil.which", lambda name: "headless")

    def run(command, **kwargs):
        root = Path(kwargs["cwd"])
        for name, content in {"solution.py": "import helper", "helper.py": "VALUE = 42", "agent.log": "secret"}.items():
            (root / name).write_text(content)
        (root / "ARTIFACTS.json").write_text('["solution.py", "helper.py"]')
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("evotoolkit.proposers.agent.subprocess.run", run)
    assert AgentProposer("codex").propose(_ctx())[0].files == {"solution.py": "import helper", "helper.py": "VALUE = 42"}


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_agent_rejects_linked_artifact(tmp_path, monkeypatch, link_type):
    secret = tmp_path / "secret.py"
    secret.write_text("secret")
    monkeypatch.setattr("evotoolkit.proposers.agent.shutil.which", lambda name: "headless")

    def run(command, **kwargs):
        artifact = Path(kwargs["cwd"]) / "solution.py"
        try:
            os.symlink(secret, artifact) if link_type == "symlink" else os.link(secret, artifact)
        except OSError as exc:
            pytest.skip(f"Filesystem does not allow {link_type}: {exc}")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("evotoolkit.proposers.agent.subprocess.run", run)
    with pytest.raises(ValueError, match="link"):
        AgentProposer("codex").propose(_ctx())


@pytest.mark.skipif(os.name != "nt", reason="Case aliases apply to Windows")
def test_chat_rejects_windows_case_aliases():
    with pytest.raises(ValueError, match="Duplicate"):
        ChatProposer(ScriptedLLM("```a.py\nx\n```\n```A.py\ny\n```")).propose(_ctx("files"))


def test_runner_attaches_chat_trace_and_links_failed_call(tmp_path):
    spec = _ctx().spec
    proposer = ChatProposer(ScriptedLLM("malformed response"))
    evaluator = SimpleNamespace(spec=spec, evaluate=lambda files: EvaluationResult(True, 1, ""))
    runner = Runner(evaluator=evaluator, proposer=proposer, strategy=EoHStrategy(spec), output_dir=tmp_path, budget=1, max_calls=1, verbose=False)
    assert proposer.eventlog is runner.eventlog
    assert runner.run() is None
    assert runner.samples == 0 and runner.failed_calls == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    failure = next(event for event in events if event["type"] == "call_failed")
    trace = json.loads((tmp_path / "traces" / f"{failure['call_id']}.json").read_text())
    assert trace["response"] == "malformed response"
