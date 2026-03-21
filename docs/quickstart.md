# Quick Start

The stable `1.0.0` workflow is:

1. Define a task.
2. Return a `TaskSpec`.
3. Wrap the task with a method interface.
4. Instantiate a method explicitly.
5. Call `run()`.

```python
from evotoolkit import EvoEngineer
from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import EvoEngineerPythonInterface, PythonTask
from evotoolkit.tools import HttpsApi


class SquareTask(PythonTask):
    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="square",
            prompt="Write a Python function `f(x)` that returns a numeric value.",
            modality="python",
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        namespace = {}
        exec(candidate_code, namespace)  # noqa: S102
        if "f" not in namespace:
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "Function `f` was not defined."})
        return EvaluationResult(valid=True, score=float(namespace["f"](3)), additional_info={})


task = SquareTask(data=None)
interface = EvoEngineerPythonInterface(task)
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key",
    model="gpt-4o",
)
algo = EvoEngineer(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=5,
)
best_solution = algo.run()
```

For string problems, use `StringTask` together with `EoHStringInterface`, `EvoEngineerStringInterface`, or `FunSearchStringInterface`.

For a runnable end-to-end example, see `examples/custom_task/my_custom_task.py` in the repository.
