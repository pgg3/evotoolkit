# Custom Task Tutorial

Learn how to build a custom `PythonTask` on top of the EvoToolkit core runtime.

## Overview

This tutorial keeps only one minimal path:

- define a `PythonTask`
- describe it with `TaskSpec`
- implement evaluation logic
- run it with a built-in method

!!! tip "Complete Example Code"
    The runnable reference example is:

    - [:material-download: my_custom_task.py](https://github.com/pgg3/evotoolkit/blob/master/examples/custom_task/my_custom_task.py)

    Run locally:
    ```bash
    cd examples/custom_task
    uv run python my_custom_task.py
    ```

## Prerequisites

- Basic Python class and inheritance knowledge
- A working LLM API endpoint and key

## Step 1: Define a PythonTask

```python
import numpy as np

from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import PythonTask


class FunctionApproximationTask(PythonTask):
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
```

## Step 2: Run It With a Built-In Method

```python
import os
import numpy as np

from evotoolkit import EvoEngineer
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi


data = np.linspace(0.0, 10.0, 50)
target = np.sin(data)

task = FunctionApproximationTask(data, target)
interface = EvoEngineerPythonInterface(task)
llm_api = HttpsApi(
    api_url=os.environ["LLM_API_URL"],
    key=os.environ["LLM_API_KEY"],
    model=os.environ.get("LLM_MODEL", "gpt-4o"),
)

algo = EvoEngineer(
    interface=interface,
    output_path="./results/custom_task_python",
    running_llm=llm_api,
    max_generations=5,
)
best_solution = algo.run()
```

## What Matters

- Use `TaskSpec.prompt` to describe the optimization target.
- Put task-specific data on the task instance in `__init__()`.
- Implement all scoring inside `_evaluate_code_impl()`.
- Return higher scores for better candidates.

## Best Practices

### Validate the candidate before scoring

Check that the expected function exists, returns the right kind of value, and does not emit `NaN` or `Inf` if your task is numeric.

### Keep evaluation deterministic

If possible, keep your evaluator stable across runs. That makes comparisons, checkpoint resume, and debugging much easier.

### Keep bootstrap logic out of the task

If you want to show the model an example implementation, put it in the method prompt or in a custom interface. The task should only define the problem and the evaluator.

## Complete Example

See `examples/custom_task/my_custom_task.py` for the full runnable example.

## Next Steps

- See `docs/extensions.md` for the current extension model
- See `docs/migration.md` if you are porting an older custom task
- See `docs/api/tasks.md` and `docs/api/interfaces.md` for low-level API details
