# Quick Start

The core `3.0.0` workflow is:

1. Define a task with the generic SDK.
2. Wrap it with a generic interface.
3. Instantiate a method explicitly.
4. Call `run()`.

```python
from evotoolkit import EvoEngineer
from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.python_task import EvoEngineerPythonInterface, PythonTask
from evotoolkit.tools import HttpsApi


class SquareTask(PythonTask):
    def _process_data(self, data):
        self.task_info = {"name": "square"}

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        namespace = {}
        exec(candidate_code, namespace)  # noqa: S102
        score = float(namespace["f"](3))
        return EvaluationResult(valid=True, score=score, additional_info={})

    def get_base_task_description(self) -> str:
        return "Write a Python function `f(x)` that returns a numeric value."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("def f(x):\n    return x")


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
result = algo.run()
```

Ready-made domains such as scientific regression and CUDA workflows now live in `evotoolkit-tasks`.
