# EvoToolkit

[![CI](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![Python](https://img.shields.io/pypi/pyversions/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docs](https://readthedocs.org/projects/evotoolkit/badge/?version=latest)](https://evotoolkit.readthedocs.io/)

**EvoToolkit** is a Python toolkit for LLM-driven evolutionary optimization. It provides a modular, three-layer architecture — **Method**, **Interface**, and **Task** — that decouples search algorithms from problem definitions, making it easy to apply evolutionary strategies to diverse domains.

## Key Features

- **Built-in evolutionary methods**: `EoH`, `EvoEngineer`, and `FunSearch`, ready to use out of the box
- **Extensible task system**: define custom optimization problems via `PythonTask` or `StringTask`
- **LLM-agnostic**: works with any OpenAI-compatible API endpoint
- **Checkpointing**: automatic state persistence and resumable runs
- **Lightweight**: minimal core dependencies (`numpy`, `scipy`)

## Installation

```bash
pip install evotoolkit
```

## Quick Start

```python
from evotoolkit import EvoEngineer
from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import EvoEngineerPythonInterface, PythonTask
from evotoolkit.tools import HttpsApi


class MyTask(PythonTask):
    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="square",
            prompt="Write a Python function `f(x)` that returns x squared.",
            modality="python",
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        namespace = {}
        exec(candidate_code, namespace)  # noqa: S102
        fn = namespace.get("f")
        if fn is None:
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={})
        score = -abs(fn(5) - 25)  # closer to 25 is better
        return EvaluationResult(valid=True, score=score, additional_info={})


task = MyTask(data=None)
interface = EvoEngineerPythonInterface(task)
llm = HttpsApi(api_url="https://api.openai.com/v1/chat/completions", key="your-key", model="gpt-4o")
algo = EvoEngineer(interface=interface, output_path="./results", running_llm=llm, max_generations=5)
best = algo.run()
```

See [`examples/custom_task/`](examples/custom_task/) for a complete runnable example.

## Documentation

Full documentation (English & Chinese) is available at [evotoolkit.readthedocs.io](https://evotoolkit.readthedocs.io/).

## License

[MIT](LICENSE)
