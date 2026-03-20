# EvoToolkit

EvoToolkit is the core SDK for LLM-driven evolutionary search over executable or structured solutions.

As of `3.0.0`, this repository intentionally focuses on the reusable runtime layer:

- evolutionary algorithms (`EoH`, `EvoEngineer`, `FunSearch`)
- explicit method lifecycle and checkpointing
- abstract task and interface contracts
- generic `PythonTask` / `StringTask` SDK layers
- generic Python / String method interfaces
- explicit algorithm instantiation and `run()`

Concrete task families no longer live in the core package. They were moved to the companion package [`evotoolkit-tasks`](../tasks/README.md).

## Install

Core SDK only:

```bash
pip install evotoolkit
```

Core SDK plus concrete task families:

```bash
pip install evotoolkit-tasks
```

## Quick Start

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

## Package Boundary

- `evotoolkit`: core SDK
- `evotoolkit-tasks`: concrete domains, hardware-backed workflows, reproducibility examples

Examples of moved task families:

- scientific regression
- prompt optimization
- adversarial attacks
- control Box2D
- CUDA engineering
- CANN init

## Development

```bash
uv sync --group dev
uv run pytest
uv run mkdocs build
uv build --out-dir dist
```

## Companion Package

The workspace directory [`../tasks`](../tasks/README.md) contains the first companion package produced by the split. It keeps the existing domain implementations intact while the core package stays small and reusable.

## Runtime Artifacts

Each run writes:

- `checkpoint/state.pkl`
- `checkpoint/manifest.json`
- readable `history/*.json`
- readable `summary/*.json`

Checkpoint restore is explicit: recreate the algorithm object, call `load_checkpoint()`, then call `run()` again.
