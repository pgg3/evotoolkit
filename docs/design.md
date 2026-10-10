# EvoToolkit 2.0 Design

Status: implemented and released as 2.0.0. Where the implementation deviates
from the original draft, the deviation is recorded in the section it belongs
to. Usage is documented in `extensions.md`.

2.0 is a clean break from 1.x: no compatibility shims, no deprecated paths.

## 1. Scope

### What EvoToolkit is

A minimal harness for LLM-driven program search. The problem shape it serves:

- an **executable artifact**, stored as a set of files (one Python module, a
  CUDA kernel plus its launcher, a prompt file, ...),
- an **evaluator** that runs the artifact and returns a score,
- a **budget** (number of proposals, plus a cap on proposer calls; see invariant 6).

Everything the toolkit does is: pick parents from an archive, ask a proposer
for new artifacts, evaluate them, record them, repeat.

### Who uses it

| User | Needs |
|---|---|
| Researcher comparing search strategies | Same loop, same budget accounting, same event log for FunSearch, EoH, EvoEngineer, and new strategies. |
| Researcher comparing proposers | API model vs coding agent (Claude Code, Codex, ...) on the same task with cost recorded. |
| Practitioner optimizing a kernel or heuristic | Write one `Evaluator`, pick a strategy, run. |
| Analyst | Read the JSONL event log; rebuild lineage, score trajectories, stagnation. |

### Usage scenarios

1. Heuristic design for combinatorial optimization. Evaluator averages gap over an instance set.
2. Kernel or operator optimization (EvoEngineer). Evaluator compiles, checks correctness, profiles.
3. Mathematical constructions. Evaluator is a checker.
4. Prompt or configuration search. Evaluator is downstream task score.
5. Later: meta-level search over the instruction templates themselves. Not built now; the design must not block it.

## 2. Structure

### Components

Five components, each answering one question. Arrows show allowed dependencies.

```
Runner  -->  Strategy  -->  Archive
  |             |
  |             +--->  (instruction templates: data, not code)
  +--->  Proposer
  +--->  Evaluator
  +--->  EventLog
```

| Component | Question it answers | Knows about | Must not know about |
|---|---|---|---|
| Evaluator | How good is this artifact? | the artifact files | proposers, archive, threads |
| Proposer | Given parents and an instruction, what are some new artifacts? | ProposalContext | archive, scoring, threads |
| Archive | What is stored and how are parents sampled? | Solutions and their scores | LLMs, prompts, files on disk |
| Strategy | This step, which parents and which instruction? | Archive, templates | LLM calls, threads, files on disk |
| Runner | Loop, concurrency, budget, persistence. | all of the above | prompt text, selection policy |

### Data objects

```python
Files = dict[str, str]          # relative path -> content

@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt: str                 # task description shown to the proposer
    modality: str               # "python" | "cuda" | "string" | ...
    main_file: str              # the file a single-file proposer writes, e.g. "solution.py"
    extras: dict[str, Any]

@dataclass
class EvaluationResult:
    valid: bool
    score: float | None         # higher is better
    feedback: str               # human/LLM readable: errors, profiler output, ...
    metrics: dict[str, Any]     # structured extras, including any objective vector
    cost_seconds: float

@dataclass(frozen=True)
class Instruction:
    name: str                   # "i1", "e1", "funsearch", "diff_refine", ...
    num_parents: int
    text: str                   # modality-agnostic; may use {modality}, {main_file}
    output_format: str          # "code_block" | "name_code_thought" | "files"

@dataclass
class ProposalContext:
    spec: TaskSpec
    parents: list["Solution"]
    instruction: Instruction
    best: "Solution | None"
    hints: list[str]            # free-form, e.g. descriptions of other solutions
    num_proposals: int          # how many the strategy wants from this call
    call_id: str                # assigned by the Runner before dispatch; proposers echo it

@dataclass
class Proposal:
    call_id: str                # shared by all proposals from one proposer call
    files: Files
    name: str
    rationale: str
    usage: dict[str, Any]       # per call, not per proposal; dedupe by call_id

@dataclass
class Solution:
    id: str                     # content hash of files, 16 hex chars
    files: Files
    name: str
    rationale: str
    parent_ids: list[str]
    instruction: str            # Instruction.name
    proposer: str               # Proposer.name
    call_id: str
    usage: dict[str, Any]
    evaluation: EvaluationResult | None
    created_step: int
```

`Solution.id = sha256("\0".join(f"{path}\0{content}" for path in sorted(files)))[:16]`.

The GA labels are gone from the data model. A crossover is an instruction
with `num_parents=2`, a mutation is an instruction with `num_parents=1`. EoH's
five operators and FunSearch's one become rows in a template table.

### Invariants

1. **Log before archive.** A Solution is appended to the event log before it is inserted into the archive. The log is append-only.
2. **State is a function of the log.** `Runner.resume()` rebuilds the archive and counters by replaying the log. No pickle.
3. **Strategy is pure.** `Strategy.next_batch()` reads the archive and returns contexts. It performs no I/O, no LLM calls, no threading. Parallelism belongs to the Runner: it tells the Strategy how many slots are free through `max_n` and calls `observe` serially, so a Strategy never needs a lock.
4. **Proposer is stateless across calls.** Any memory lives in the archive or the log.
5. **Content-addressed identity.** Two solutions with identical files have identical `id`. Archives may reject duplicates; the log still records the attempt.
6. **Budget is counted in the Runner only, in proposals.** Strategies see `remaining_budget` as a number; they never decrement it. A failed proposer call yields zero proposals and does not consume proposal budget; a separate `max_calls` cap bounds total calls.
7. **The archive is the only shared channel between proposals.** Any dependency that does not go through the archive is private to some Proposer and lives in its trace. Dependencies across an evaluation are expressed by the Strategy through `observe` and the next `next_batch`; chained proposers compose into one Proposer. The core does not model a dependency graph between proposals.
8. **Evaluators never raise.** Failure is `valid=False` with `feedback` set. Isolation and timeouts are the Evaluator's job; the Runner only measures `cost_seconds` and still catches as a last resort.

## 3. Interface contracts

All contracts are `typing.Protocol`. Implementations do not inherit.

```python
class Evaluator(Protocol):
    spec: TaskSpec
    def evaluate(self, files: Files) -> EvaluationResult: ...

class Proposer(Protocol):
    name: str
    def propose(self, ctx: ProposalContext) -> list[Proposal]: ...   # may raise

class Archive(Protocol):
    def add(self, solution: Solution) -> None: ...
    def sample(self, n: int, *, rng: np.random.Generator) -> list[Solution]: ...
    def best(self) -> Solution | None: ...
    def __len__(self) -> int: ...

class Strategy(Protocol):
    name: str
    archive: Archive
    def next_batch(self, step: int, max_n: int, remaining_budget: int, rng: np.random.Generator) -> list[ProposalContext]: ...
    def observe(self, solutions: list[Solution]) -> None: ...   # add to archive, trim, etc.
    def should_stop(self, step: int) -> bool: ...
```

Runner is concrete, not a protocol:

```python
class Runner:
    def __init__(self, *, evaluator, proposer, strategy, output_dir, budget,
                 max_calls=None, propose_workers=4, eval_workers=4,
                 mode="generational", seed=0, verbose=True): ...
    def run(self) -> Solution | None: ...
    def step(self) -> None: ...
    @classmethod
    def resume(cls, output_dir, *, evaluator, proposer, strategy) -> "Runner": ...
```

`Runner.step()` does exactly: `ctxs = strategy.next_batch(...)`, push them
through the two-stage pipeline, build Solutions, append each to the log,
`strategy.observe(solutions)`, bump counters, write checkpoint marker.

Two pipeline stages with independent parallelism:

1. **Propose stage**: `propose_workers` threads call `proposer.propose(ctx)` concurrently; each call returns a list. An exception becomes a `call_failed` event.
2. **Evaluate stage**: every returned list is split immediately and each item is submitted to `eval_workers` threads running `evaluator.evaluate`, without waiting for sibling calls.

Two scheduling modes run the same Strategy unchanged:

- `generational`: dispatch a batch, wait for all of it, ask for the next. Behavior of EoH and EvoEngineer; the golden baseline is recorded in this mode.
- `steady`: whenever a worker frees up, call `next_batch(max_n=1)` immediately. Behavior of the original FunSearch and AlphaEvolve. Higher throughput when LLM latency varies or evaluation is fast.

### Built-in Proposers

- `ChatProposer(llm, templates)`: formats `ctx` into messages using the
  instruction's `text` and `output_format`, calls the LLM, parses by
  `output_format` into `Files`. `code_block` and `name_code_thought` write
  `spec.main_file`; `files` parses several fenced blocks tagged with paths.
  Absorbs today's six interface files into one template table.
- `AgentProposer(backend, workdir_factory)`: materializes a workspace
  (`TASK.md` describing every parent, the last parent's files, and
  `evaluate.py` only when a `workdir_factory` or `evaluation_command` supplies
  one), runs a headless coding agent through `headless-cli`, reads back the
  main file and parent paths plus anything declared in `ARTIFACTS.json`. The
  transcript is the trace.

### Built-in Evaluator helpers

- `subprocess_evaluate(files, entry, timeout)`: writes files to a temp dir,
  runs `entry` in a child process with a timeout, returns stdout/stderr and
  exit code. Task families opt in; the core does not force isolation.

### Built-in Archives

- `RankedPopulation(max_size)`: 1.x `PopulationState` plus rank-based
  sampling. Deviation: invalid solutions are never stored; 1.x kept them as
  filler when the population had spare slots.
- `IslandArchive(num_islands, ...)`: 1.x `ProgramsDatabase` with two
  deliberate changes so that `resume` can rebuild it from the log alone. New
  solutions route to an island by content hash instead of a random draw, and
  weak islands reset after every `reset_interval` accepted unique solutions
  instead of on a wall-clock timer.

### Built-in Strategies

- `FunSearchStrategy`: one instruction, `num_parents` 2, IslandArchive.
- `EoHStrategy`: instructions i1, e1, e2, m1, m2 keeping the original
  operator semantics, rewritten onto the shared template (which also adds
  scores, feedback and the current best), RankedPopulation. Deviation: 1.x ran every operator over the whole
  population each generation; 2.0 cycles the operators across the slots the
  Runner offers, so one call to `next_batch(max_n=4)` yields e1, e2, m1, m2.
- `EvoEngineerStrategy`: init, crossover, mutation with best-so-far and hints, RankedPopulation.

Each is well under 150 lines.

## 4. Persistence layout (v1)

```
<output_dir>/
  events.jsonl            append-only, one JSON object per line
  traces/<call_id>.json   full prompt messages and raw response, or agent transcript
```

Event types:

```json
{"v": 1, "type": "run_start", "ts": "...", "task": {TaskSpec}, "strategy": "eoh", "proposer": "chat:gpt-4o", "seed": 0, "budget": 45, "config": {"max_calls": 135, "propose_workers": 4, "eval_workers": 4, "mode": "generational", "verbose": true}, "env": {"python": "3.12.1", "platform": "linux", "evotoolkit": "2.0.0"}}
{"v": 1, "type": "call_started", "ts": "...", "step": 3, "call_id": "...", "instruction": "e1", "rng_state": {...}}
{"v": 1, "type": "solution", "ts": "...", "step": 3, "solution": {Solution}}
{"v": 1, "type": "call_failed", "ts": "...", "step": 3, "call_id": "...", "instruction": "e1", "error": "..."}
{"v": 1, "type": "step_end", "ts": "...", "step": 3, "archive_size": 5, "best_id": "ab12...", "best_score": 0.91, "proposals": 4, "valid": 3, "calls": 2, "failed_calls": 0, "rng_state": {...}}
{"v": 1, "type": "run_end", "ts": "...", "status": "completed", "steps": 10, "samples": 45}
```

`call_started` and the `config` and `rng_state` fields were added during
implementation. `call_started` lets `resume` count calls that never produced a
result, so `max_calls` survives a crash. `rng_state` is the numpy bit-generator
state taken before each call and at each step end, so parent selection after a
resume continues the original random stream instead of restarting it.

Rules: `v` is bumped on any breaking field change. Readers must ignore
unknown fields. `Solution.files` is stored in full in the log; `feedback` is
truncated to 4 kB by the Runner before the solution reaches the archive, so the
live archive, the prompts built from it, and a resumed archive are identical
(the log writer keeps a `feedback_truncated: true` guard for foreign payloads).
`run_start.config.strategy` records the strategy hyperparameters and `resume`
refuses a strategy whose `config` differs. One `call_id` is minted by the
Runner per call and shared by `call_started`, `call_failed`, every resulting
`solution`, and the trace file. Traces are never
truncated. An unterminated final line, left by an interrupted process, is
skipped by readers and removed the next time the log is opened for writing;
any other malformed line is an error.

## 5. Behavioral baseline

`tests/golden/` records, for each built-in strategy with a scripted LLM and a
fixed seed: per step, the sorted multiset of `(main file content, valid,
score, parent ids)`, the instructions issued, the best score, and the failed-call count;
plus final sample, call, and status counters.

The original plan was to record this against 1.x and require 2.0 to reproduce
it. That turned out to be impossible by construction: invariant 6 says an
unparseable response is a failed call that does not spend budget, whereas 1.x
counted it as an invalid sample, so step boundaries and sample counts differ
for the same scripted responses. The 1.x snapshots were therefore retired (they
remain in git history before the rewrite) and `*-v2.json` was recorded against
2.0. The weaker evidence that was checked by hand: for all three strategies the
set of valid programs produced and the final best score are identical between
the 1.x and 2.0 runs.

The v2 files now guard against unintended drift in 2.0, not against drift from
1.x. Re-record them only with a deliberate behavior change, and say so in the
commit.

Known ceiling: thread completion order is not deterministic, so the baseline
records per-step multisets rather than sequences. Parent selection happens on
the main thread before dispatch, and generational batches are handed to
`observe` sorted by id, so archive state and selection are reproducible from
the seed (the parent-id column found that island seeding depended on thread
completion order before this ordering existed).

## 6. Deferred decisions

Considered, not built, revisit when a real need appears. Each item names its trigger.

- **Proposal dependency graph.** See invariant 7. Trigger: with agents wired in, a dependency shape appears that is neither inside one call nor through the archive.
- **Baseline for steady mode.** Results depend on timing, so golden comparison does not apply; only invariants are checked (log solution count equals budget, every parent id appears earlier).
- **Multi-objective selection.** `score` stays a single float, higher is better; objective vectors go in `metrics`. Multi-objective is a new Archive implementation that samples on `metrics`. No core change needed. Trigger: a task family and a strategy both need it.
- **Meta-level search.** Instruction templates are data and serialize to files, so a template table can be a Solution. The Evaluator runs an inner `Runner` on a small budget and scores the inner best. Needs nothing from the core beyond what exists. Trigger: someone actually does scenario 5.

## 7. Decisions

**Python, not Node or Rust.** Evaluators live in the Python ecosystem (numpy,
torch, triton, OR instance libraries). The bottleneck is LLM latency and
evaluator runtime, not the harness. Rejected: a Rust core with Python
bindings, because it buys nothing measurable and doubles the build surface.

**Artifacts are file sets, even for chat proposers.** A kernel is a kernel
plus a launcher; an agent edits files. A single string would force a second
identity scheme later. Single-file tasks set `main_file` and never notice.

**JSONL event log plus trace files, not pickle.** Pickle ties checkpoints to
class layouts and is unreadable by analysis tools. Full prompts go to
`traces/` so the log stays small enough to grep. Rejected: SQLite, because a
single-writer append-only file is simpler and diffable.

**Failed calls do not consume proposal budget.** Budget is a research
variable; a flaky API should not silently shrink it. `max_calls` prevents
infinite loops.

**Isolation is the Evaluator's job.** The core offers `subprocess_evaluate`
but does not force it. Task families know their own trust requirements.

**Protocol, not ABC.** Users should be able to pass any object with the right
methods. Rejected: ABC inheritance, because it forces users to import our base
classes and couples them to our constructor signatures.

**headless-cli subprocess for agents, not per-agent adapters.** One dependency
already normalizes Claude Code, Codex, Cursor, Gemini, OpenCode. Rejected:
writing our own adapters, which is maintenance with no research value.

**Instructions are data and modality-agnostic.** Prompt templates are rows,
not subclasses; fences and parsing live in `output_format`. Six files become
one table. This makes EoH's operators and FunSearch's prompt comparable and
makes the meta level (scenario 5) a matter of evolving rows.

**`output_format` is a string enum.** Parsers are looked up by name inside
`ChatProposer`. Rejected: attaching parser callables to `Instruction`, because
templates must serialize to JSON or YAML.

**No 1.x compatibility.** Clean break, one release. Rejected: shims and
deprecation cycles, because the user base is small and the old surface is the
thing being replaced.

**No biological vocabulary beyond "evolution".** Proposer, Proposal, Archive,
Strategy, Evaluator, Instruction. Rejected: gene, capsule, organism, crossover,
mutation as core names.
