# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Runnable PythonTask example for the EvoToolkit core package."""

import os

import numpy as np

from evotoolkit import EvoEngineer
from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import EvoEngineerPythonInterface, PythonTask
from evotoolkit.tools import HttpsApi


class FunctionApproximationTask(PythonTask):
    """Optimize a Python function against numeric targets."""

    def __init__(self, data, target, timeout_seconds=30.0):
        self.target = np.asarray(target, dtype=float)
        super().__init__(np.asarray(data, dtype=float), timeout_seconds=timeout_seconds)

    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="function_approximation",
            prompt=(
                "Write a Python function `my_function(x)` that maps a scalar input to a scalar output. "
                "The goal is to match the hidden target function as closely as possible."
            ),
            modality="python",
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        try:
            namespace = {"np": np}
            exec(candidate_code, namespace)  # noqa: S102
            if "my_function" not in namespace:
                return EvaluationResult(
                    valid=False,
                    score=float("-inf"),
                    additional_info={"error": "Function `my_function` was not defined."},
                )

            predictions = np.array([namespace["my_function"](x) for x in self.data], dtype=float)
            mse = float(np.mean((predictions - self.target) ** 2))
            return EvaluationResult(valid=True, score=-mse, additional_info={"mse": mse})
        except Exception as exc:
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": str(exc)},
            )


def build_llm_api() -> HttpsApi:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY before running this example.")

    return HttpsApi(
        api_url=os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        key=api_key,
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
    )


def run_example() -> None:
    data = np.linspace(0.0, 10.0, 50)
    target = np.sin(data)

    task = FunctionApproximationTask(data, target)
    interface = EvoEngineerPythonInterface(task)
    algo = EvoEngineer(
        interface=interface,
        output_path="./results/custom_task_python",
        running_llm=build_llm_api(),
        max_generations=5,
    )
    result = algo.run()

    print(f"Best score: {result.evaluation_res.score:.4f}")
    print(f"Best MSE: {result.evaluation_res.additional_info['mse']:.4f}")
    print(result.sol_string)


if __name__ == "__main__":
    run_example()
