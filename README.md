# EvoToolkit

EvoToolkit is the stable core SDK for LLM-driven evolutionary search over executable or structured solutions.

`1.0.0` is the first stable release of the standalone runtime. The package intentionally ships only reusable building blocks:

- evolutionary algorithms: `EoH`, `EvoEngineer`, `FunSearch`
- runtime lifecycle bases: `Method`, `IterativeMethod`, `PopulationMethod`
- checkpointing and readable artifacts through `RunStore`
- generic `PythonTask` and `StringTask` SDK layers
- generic Python and String interfaces for the built-in methods

Concrete domain tasks should live in external packages or in your own repository on top of this core.

## Install

```bash
pip install evotoolkit
```

## Quick Start

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
result = algo.run()
```

## Building On Top

The intended extension workflow is explicit:

1. Define a `PythonTask` or `StringTask`.
2. Return a `TaskSpec` from `build_python_spec()` or `build_string_spec()`.
3. Pair the task with a generic interface such as `EvoEngineerPythonInterface`.
4. Instantiate a method class directly and call `run()`.

If you need a domain package, keep those concrete tasks outside `src/evotoolkit` and expose them through your own package imports.

A runnable reference implementation lives in `examples/custom_task/my_custom_task.py`.

## Development

```bash
uv sync --group dev
uv run pytest
uv run mkdocs build
uv build --out-dir dist
```

## Runtime Artifacts

Each run writes:

- `checkpoint/state.pkl`
- `checkpoint/manifest.json`
- readable `history/*.json`
- readable `summary/*.json`

Checkpoint restore is explicit: recreate the algorithm object, call `load_checkpoint()`, then call `run()` again.
