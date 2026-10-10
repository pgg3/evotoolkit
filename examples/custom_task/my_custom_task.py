"""Run with --offline for a local smoke test, or configure LLM_API_KEY."""

import argparse
import json
import math
import os
import threading
from pathlib import Path

from evotoolkit import EvaluationResult, Runner, TaskSpec
from evotoolkit.evaluators import subprocess_evaluate
from evotoolkit.proposers import ChatProposer
from evotoolkit.strategies import EvoEngineerStrategy
from evotoolkit.tools import HttpsApi


class FunctionApproximationEvaluator:
    spec = TaskSpec(
        name="function_approximation",
        prompt="Write solution.py defining my_function(x), approximating sin(x) for 0 <= x <= 3. Use the Python standard library.",
        modality="python",
        main_file="solution.py",
    )

    def evaluate(self, files):
        scorer = (
            "import json, math\nfrom solution import my_function\n"
            "xs = [i / 10 for i in range(31)]\n"
            "mse = sum((my_function(x) - math.sin(x)) ** 2 for x in xs) / len(xs)\n"
            "print(json.dumps({'mse': mse}))\n"
        )
        result = subprocess_evaluate(files | {"_score.py": scorer}, "_score.py", timeout=5)
        if result.returncode:
            return EvaluationResult(False, None, result.stderr)
        try:
            mse = float(json.loads(result.stdout.splitlines()[-1])["mse"])
            if not math.isfinite(mse):
                raise ValueError("Non-finite MSE")
            return EvaluationResult(True, -mse, "", metrics={"mse": mse})
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return EvaluationResult(False, None, str(exc))


class ScriptedLLM:
    """Deterministic local responses; no API calls."""

    def __init__(self):
        self._index = 0
        self._lock = threading.Lock()

    def get_response(self, messages):
        expressions = ["0", "x", "x - x**3 / 6", "math.sin(x)"]
        with self._lock:
            expression = expressions[self._index % len(expressions)]
            self._index += 1
        return f"name: approximation\n```python\nimport math\ndef my_function(x):\n    return {expression}\n```\nthought: Try a candidate.", {}


def build_llm_api():
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("Set LLM_API_KEY or pass --offline.")
    return HttpsApi(api_url=os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"), key=key, model=os.environ.get("LLM_MODEL", "gpt-4o"))


def run_example(*, output_dir, offline=False, mode="generational", resume=False):
    evaluator = FunctionApproximationEvaluator()
    proposer = ChatProposer(ScriptedLLM() if offline else build_llm_api())
    strategy = EvoEngineerStrategy(evaluator.spec)
    if resume:
        runner = Runner.resume(output_dir, evaluator=evaluator, proposer=proposer, strategy=strategy)
    else:
        runner = Runner(evaluator=evaluator, proposer=proposer, strategy=strategy, output_dir=output_dir, budget=12, mode=mode, seed=42)
    best = runner.run()
    if best is None:
        print("No valid solution; inspect events.jsonl and traces/.")
    else:
        print(f"Best score: {best.score:.6f}")
        print(best.files[evaluator.spec.main_file])
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=["generational", "steady"], default="generational")
    parser.add_argument("--output-dir", type=Path, default=Path("results/custom_task"))
    args = parser.parse_args()
    run_example(output_dir=args.output_dir, offline=args.offline, mode=args.mode, resume=args.resume)
