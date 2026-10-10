# Extensions and API

## Contracts

Implement objects with the following members; inheritance is unnecessary. Types live in `evotoolkit.core`.

| Component | Contract |
|---|---|
| Evaluator | `spec: TaskSpec`, `evaluate(files) -> EvaluationResult` |
| Proposer | `name: str`, `propose(ctx: ProposalContext) -> list[Proposal]` |
| Archive | `add(solution)`, `sample(n, *, rng)`, `best()`, `len(archive)` |
| Strategy | `name`, `archive`, `next_batch(step, max_n, remaining_budget, rng)`, `observe(solutions)`, `should_stop(step)` |

`Files` is `dict[str, str]`: relative paths with `/` separators to text contents. `TaskSpec` has name, prompt, modality, main_file, and optional extras. `Instruction` has name, num_parents, text, output_format. Templates interpolate `{modality}` and `{main_file}`; escape other literal braces as `{{` and `}}`.

`ProposalContext` contains spec, parents, instruction, best, hints and num_proposals. Return no more than the requested count. Every `Proposal` must echo `ctx.call_id` (the Runner mints it before dispatch and rejects proposals carrying any other ID); proposals from one call share it and the usage dictionary. A `Solution` adds content hash identity, parent IDs, proposer and instruction names, evaluation and created_step. Use `Solution.create(...)` for new solutions and `to_dict`/`from_dict` for serialization; unknown fields are ignored when reading.

Strategy code runs serially on the Runner thread; proposal and evaluation calls run concurrently. Use the supplied RNG for selection. Keep replayable state in the archive and `observe`, not in `next_batch`. Evaluators must enforce their own timeouts and return invalid results on failure; Runner catches exceptions as a fallback. A valid score must be finite. Put extra objectives in `metrics`.

## Built-ins

- `RankedPopulation(max_size=5)`: finite valid unique solutions, descending scores and rank-weighted sampling with replacement.
- `IslandArchive(num_islands=5, reset_interval=30000, ...)`: score clusters, temperature-weighted sampling, shorter artifacts within clusters. The first valid candidate seeds every island. Content hashes route later candidates; accepted unique solution counts trigger weak-island resets. These deterministic rules replace v1's sampled-island routing and wall-clock resets so replay reconstructs the archive.
- `EvoEngineerStrategy(spec, pop_size=5)`: init, crossover and mutation, best candidate and rationale hints.
- `EoHStrategy(spec, pop_size=5, selection_num=2)`: i1, e1, e2, m1, m2 instructions.
- `FunSearchStrategy(spec, num_islands=5, num_parents=2, reset_interval=30000)`: island sampling with one improvement instruction.

The built-ins stop on Runner's budget/call cap. For a custom stop criterion implement `should_stop(step)` in your strategy.

## Chat proposer

`ChatProposer(llm, templates=None, *, eventlog=None)` uses `llm.get_response(messages) -> (text, usage)`. Pass a mapping of instruction names to `Instruction` objects to override templates. Defaults are data in `evotoolkit.proposers.templates.INSTRUCTIONS`.

Output formats:

- `code_block`: exactly one fenced block, written to `spec.main_file`; optional leading `{rationale}`.
- `name_code_thought`: `name:` before one code fence, then `thought:` after it.
- `files`: multiple fences with a relative path directly after the opening fence, such as ` ```src/main.py `.

Multiple alternatives are separated by a line containing only `===PROPOSAL===`. The number must equal `ctx.num_proposals`. Parse errors are failed calls, preserved in full traces. Runner attaches its EventLog to built-in proposers automatically.

## Agent proposer

`AgentProposer(backend, workdir_factory=None, *, timeout=600, eventlog=None)` requires the separate executable installed by:

```bash
npm install -g @roberttlange/headless
```

Authenticate your chosen backend separately. The adapter uses `headless <backend> --prompt-file TASK.md --work-dir DIR --json --usage --timeout N`, as specified by the [upstream CLI documentation](https://github.com/RobertTLange/headless-cli/blob/main/docs/usage.md).

Each call is a fresh workspace and returns one candidate; request `num_proposals=1`. The last parent's files seed the workspace, while TASK.md describes every parent and the instruction. A custom `workdir_factory(ctx)` returns a context manager yielding a dedicated directory; it may provide `evaluate.py` and support files. Alternatively `spec.extras['evaluation_command']` can contain an argv list used to generate `evaluate.py`. Without either, evaluation occurs after proposal generation.

TASK.md, evaluate.py and factory support files are harness inputs. Returned artifacts default to the main file and existing parent artifact paths. To add or remove files, the agent writes a JSON array of relative output paths to `ARTIFACTS.json`; this manifest is not itself an artifact. This keeps unrelated workspace outputs out of the proposal.

Headless uses backend auto-approval. Temporary workspaces and subprocesses are not security sandboxes; provide OS/container isolation appropriate to the generated code. The adapter does not install agents or manage credentials.

## Evaluating programs

`from evotoolkit.evaluators import subprocess_evaluate` provides an opt-in helper:

```python
result = subprocess_evaluate({"main.py": "print(42)"}, "main.py", timeout=5)
assert result.returncode == 0
assert result.stdout.strip() == "42"
```

A string entry is a relative Python filename; an argv list runs an explicit command. The helper writes validated relative paths into a temporary directory and returns `subprocess.CompletedProcess`. Timeout uses return code 124; setup failures use 1. stdout/stderr are preserved. Timeout terminates the direct child; process-tree containment remains the evaluator's responsibility.

## Event log

`EventLog` supports run_start, call_started, solution, call_failed, step_end and run_end. An unterminated final line left by an interrupted process is skipped by readers and removed when the log is next opened for writing; any other malformed line raises an error. Readers ignore unknown fields. Writes are flushed but not fsynced. Do not run two writers against one output directory.

Resume checks the task, component names and the strategy's `config` (its constructor hyperparameters); callers must still supply the original proposer configuration. Logged solutions rebuild state; results that never reached the log are lost. RNG snapshots retain selection progress, but live scheduling and external model responses remain nondeterministic.
