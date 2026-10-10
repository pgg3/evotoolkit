# Quick start

From a checkout, run `uv sync --group dev`, then a deterministic local search:

```bash
uv run python examples/custom_task/my_custom_task.py --offline --output-dir results/demo
uv run python examples/custom_task/my_custom_task.py --offline --mode steady --output-dir results/steady
```

The example supplies twelve proposals, scores a Python function against sin(x), and prints the best artifact. Set `LLM_API_KEY` (plus optional `LLM_API_URL`, `LLM_MODEL`) and omit `--offline` for live calls.

## Compose a search

```python
from evotoolkit import EvaluationResult, Runner, TaskSpec
from evotoolkit.proposers import ChatProposer
from evotoolkit.strategies import EoHStrategy

class GreetingEvaluator:
    spec = TaskSpec("greeting", "Write hello in answer.txt.", "string", "answer.txt")

    def evaluate(self, files):
        return EvaluationResult(True, float(files.get("answer.txt", "").strip() == "hello"), "")

# llm.get_response(messages) returns (response_text, usage_dict).
evaluator = GreetingEvaluator()
proposer = ChatProposer(llm)
strategy = EoHStrategy(evaluator.spec)
runner = Runner(evaluator=evaluator, proposer=proposer, strategy=strategy,
                output_dir="results/greeting", budget=20, seed=42,
                propose_workers=4, eval_workers=4, mode="generational")
best = runner.run()
```

Replace the strategy with `EvoEngineerStrategy(spec)` or `FunSearchStrategy(spec)` without changing the loop. Scores are scalar and higher is better. Invalid results use `EvaluationResult(False, None, feedback)`.

## Budgets and concurrency

`budget` counts evaluated proposals, including invalid and duplicate artifacts. Failed/empty/malformed calls produce `call_failed` events without consuming proposal budget. `max_calls` bounds all dispatched calls, including calls interrupted by a crash; its default is `max(1, 3 * budget)`. Each context reserves its requested proposals, so concurrent calls cannot exceed the remaining budget. Returning more than requested rejects the entire call.

`generational` selects a batch from the current archive and observes its results before the next batch. `steady` refills available proposer workers continuously and observes each completed evaluation. Both feed evaluation workers immediately when an individual proposer call finishes. `step()` executes one bounded batch; `run()` enables continuous steady scheduling.

`runner.samples`, `calls`, `failed_calls`, `steps`, `status`, and `best_solution` expose progress. Final status is `completed` when the proposal budget is used, `max_calls` when the call cap is reached, or `stopped` when the strategy stops or yields no work. `run()` closes the log; manual stepping should use a context manager or call `close()`.

## Resume

```python
runner = Runner.resume("results/greeting", evaluator=evaluator,
                       proposer=ChatProposer(llm), strategy=EoHStrategy(evaluator.spec))
best = runner.run()
```

Supply a fresh empty strategy with the same parameters. Task specification and component names must match. Budget, scheduling, worker counts and RNG state come from the event log. Completed solutions rebuild the archive; unfinished calls consume the call cap but not proposal budget. Resume cannot restore an external model's hidden state or guarantee identical completion order.

`events.jsonl` contains full artifact files, evaluations, parent IDs, counters and checkpoint RNG state. `traces/<call_id>.json` contains full prompts, responses or agent transcripts. Feedback in solution events is limited to 4 KiB; traces are untruncated. Call usage is per call: deduplicate `Solution.call_id` when totaling tokens.
