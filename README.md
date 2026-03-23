# EvoToolkit Core

EvoToolkit is the core SDK for LLM-driven evolutionary search over executable or structured solutions.

This branch prepares `1.0.1rc1`, a release candidate built on top of the `v1.0.0` stable runtime line. The package intentionally ships only reusable building blocks:

- built-in methods: `EoH`, `EvoEngineer`, `FunSearch`
- runtime lifecycle bases: `Method`, `IterativeMethod`, `PopulationMethod`
- checkpointing and readable artifacts through `RunStore`
- generic `PythonTask` and `StringTask` SDK layers
- generic Python and string interfaces for the built-in methods
- OpenAI-compatible HTTP client utilities in `evotoolkit.tools`

Concrete domain tasks, hardware-backed workflows, and application-specific examples should live in your own package or repository on top of this core.

## Install

```bash
pip install evotoolkit
```

To test release candidates after they are published:

```bash
pip install --pre evotoolkit
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
best_solution = algo.run()
```

## Documentation

The published documentation mirrors the `docs/` directory and keeps the core pages in English and Chinese:

- `index`
- `installation`
- `quickstart`
- `extensions`
- `migration`

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mkdocs build
uv build --out-dir dist
```

The runnable repository example lives in `examples/custom_task/my_custom_task.py`.

## Runtime Artifacts

Each run writes:

- `checkpoint/state.pkl`
- `checkpoint/manifest.json`
- readable `history/*.json`
- readable `summary/*.json`

Checkpoint restore is explicit: recreate the algorithm object, call `load_checkpoint()`, then call `run()` again.
