"""Generational-mode snapshots for the three built-in strategies (design.md section 6).

Record: python -m tests.golden.golden record
Check:  pytest tests/golden

These guard 2.0 against unintended drift. The 1.x snapshots were retired because
invariant 6 (failed calls do not spend budget) makes sample counts differ by
construction; see design.md section 6. Re-record only with a deliberate behavior
change, and say so in the commit.
"""

import hashlib
import json
import sys
import tempfile
import threading
from pathlib import Path

from evotoolkit.core import EvaluationResult, Runner, TaskSpec
from evotoolkit.proposers.chat import ChatProposer
from evotoolkit.strategies import EoHStrategy, EvoEngineerStrategy, FunSearchStrategy

GOLDEN_DIR = Path(__file__).parent
SEED = 1234
_VARIANTS = [
    "def f(x):\n    return x * 2",
    "def f(x):\n    return x * 3",
    "def f(x):\n    return x + 10",
    "def g(x):\n    return x",
    "def f(x):\n    return x ** 2 + 1",
    None,
    "def f(x):\n    return x / 2",
    "def f(x):\n    return 7",
]


class ScriptedLLM:
    def __init__(self):
        self._seen = {}
        self._lock = threading.Lock()

    def get_response(self, messages):
        # Reply is a function of the prompt plus how often that exact prompt was seen, so thread timing cannot
        # change which context gets which reply (identical prompts are interchangeable).
        prompt = messages[0]["content"]
        with self._lock:
            repeat = self._seen[prompt] = self._seen.get(prompt, 0) + 1
        index = (int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16) + repeat) % len(_VARIANTS)
        code = _VARIANTS[index]
        response = f"name: candidate_{index}\ncode:\n```python\n{code}\n```\nthought: scripted idea" if code is not None else "no code here at all"
        return response, {"prompt_tokens": 10, "completion_tokens": 5}


class SquareEvaluator:
    spec = TaskSpec("square", "Write f(x) close to x squared.", "python", "solution.py")

    def evaluate(self, files):
        namespace = {}
        try:
            exec(files["solution.py"], namespace)
            if "f" not in namespace:
                return EvaluationResult(False, None, "no f")
            return EvaluationResult(True, -abs(namespace["f"](5) - 25), "")
        except Exception as exc:
            return EvaluationResult(False, None, str(exc))


RUNS = {"eoh": EoHStrategy, "evoengineer": EvoEngineerStrategy, "funsearch": FunSearchStrategy}


def run_search(name, out):
    evaluator = SquareEvaluator()
    kwargs = {"num_islands": 3} if name == "funsearch" else {"pop_size": 4}
    runner = Runner(
        evaluator=evaluator,
        proposer=ChatProposer(ScriptedLLM()),
        strategy=RUNS[name](evaluator.spec, **kwargs),
        output_dir=out,
        budget=12,
        propose_workers=4,
        eval_workers=2,
        mode="generational",
        seed=SEED,
        verbose=False,
    )
    runner.run()
    events = list(runner.eventlog.events())
    seen = set()
    for event in events:
        if event["type"] == "solution":
            candidate = event["solution"]
            assert set(candidate["parent_ids"]) <= seen
            seen.add(candidate["id"])
    steps = []
    for marker in (event for event in events if event["type"] == "step_end"):
        rows = []
        for event in events:
            if event["type"] == "solution" and event["step"] == marker["step"]:
                candidate = event["solution"]
                evaluation = candidate["evaluation"]
                rows.append([candidate["files"]["solution.py"], evaluation["valid"], evaluation["score"], sorted(candidate["parent_ids"])])
        instructions = sorted(event["instruction"] for event in events if event["type"] == "call_started" and event["step"] == marker["step"])
        steps.append({"new": sorted(rows, key=json.dumps), "instructions": instructions, "best": marker["best_score"], "failed_calls": marker["failed_calls"]})
    assert sum(len(step["new"]) for step in steps) == runner.budget
    return {"steps": steps, "samples": runner.samples, "calls": runner.calls, "failed_calls": runner.failed_calls, "status": runner.status}


def produce(name):
    with tempfile.TemporaryDirectory() as out:
        return run_search(name, out)


def golden_path(name):
    return GOLDEN_DIR / f"{name}-v2.json"


def record():
    for name in RUNS:
        golden_path(name).write_text(json.dumps(produce(name), indent=2) + "\n", encoding="utf-8")
        print(f"recorded {golden_path(name)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        record()
    else:
        print(__doc__)
