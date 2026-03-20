# Migration To 3.0

`3.0.0` is a breaking release.

Key changes:

- `evotoolkit` keeps only the reusable SDK
- concrete task imports move to `evotoolkit_tasks...`
- hardware-backed workflows no longer ship with the core package
- `evotoolkit.solve(...)` is removed; instantiate a method explicitly and call `run()`

Examples:

```python
# old
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask

# new
from evotoolkit_tasks.python_task.scientific_regression import ScientificRegressionTask
```

Generic SDK imports stay in the core package:

```python
from evotoolkit.task.python_task import PythonTask, EvoEngineerPythonInterface
from evotoolkit.task.string_optimization import StringTask, EoHStringInterface
```

Low-level runtime usage is now explicit:

```python
from evotoolkit import EvoEngineer

algo = EvoEngineer(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=5,
)
result = algo.run()
```
