# Migration From Pre-1.0 Releases

`v1.0.1` is the current stable release of the standalone EvoToolkit core. The `1.0.0` release established the stable explicit-runtime surface, and `1.0.1` keeps that runtime unchanged.

If you used earlier pre-`1.0` releases or legacy APIs, these are the important changes:

- `evotoolkit.solve(...)` is gone; instantiate a method class directly and call `run()`
- the core package no longer ships concrete domain tasks
- task classes now describe themselves through `TaskSpec`
- interfaces do not build initial solutions anymore
- initialization is owned by each concrete method or its prompting strategy

## Task API Mapping

If you have custom tasks written against the older prerelease API, migrate them like this:

- `get_base_task_description()` -> `TaskSpec.prompt`
- `_process_data()` -> plain `__init__()` state plus `build_*_spec()`
- `evaluate_code(...)` / `evaluate_string(...)` helpers -> `evaluate()` or `_evaluate_*_impl()`

Example:

```python
from evotoolkit.core import TaskSpec
from evotoolkit.task.python_task import PythonTask


class MyTask(PythonTask):
    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="my_task",
            prompt="Describe the optimization target here.",
            modality="python",
        )
```

If you previously returned a baseline candidate from the task layer, remove that hook. Put any bootstrap examples directly into method prompts or into your custom interface logic instead.

## Runtime Usage Mapping

Old:

```python
result = evotoolkit.solve(interface=interface, output_path="./results", running_llm=llm_api)
```

New:

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

If you previously relied on built-in domain tasks from earlier prerelease branches, move them into your own package first, then reintroduce them on top of the current core runtime.
