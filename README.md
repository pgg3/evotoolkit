# EvoToolkit

[![CI](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pgg3/evotoolkit/branch/master/graph/badge.svg)](https://codecov.io/gh/pgg3/evotoolkit)
[![PyPI](https://img.shields.io/pypi/v/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![Python](https://img.shields.io/pypi/pyversions/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LLM-driven evolutionary framework for executable solutions**

EvoToolkit is a Python framework for combining large language models with evolutionary search over executable text. Its core software contribution is a reusable `Method -> Interface -> Task` architecture plus a shared execution substrate for prompting, parsing, evaluation history, and algorithm dispatch.

The repository also ships built-in **reference task families** that demonstrate portability across distinct solution spaces, together with `examples/` scripts that serve as tutorial and reproducibility assets. Those examples are not the core framework itself.

## Installation

Core installation:

```bash
pip install evotoolkit
```

Optional reference-task extras:

```bash
pip install "evotoolkit[scientific_regression]"
pip install "evotoolkit[prompt_engineering]"
pip install "evotoolkit[adversarial_attack]"
pip install "evotoolkit[control_box2d]"
pip install "evotoolkit[cuda_engineering]"
pip install "evotoolkit[cann_init]"
pip install "evotoolkit[all_tasks]"
```

The published package is tested on Python 3.10-3.12. CUDA and CANN workflows remain optional hardware-backed extensions and require the relevant external toolchains in addition to Python extras.

## Framework Capabilities

- Unified top-level API via `evotoolkit.solve(...)`
- Reusable `Method`, `Interface`, and `Task` abstractions
- Shared run-state, history, summary, and output management
- OpenAI-compatible LLM client utilities
- Public packaging, CI, docs, and release artifacts

## Supported Algorithms

| Algorithm | Description |
|---|---|
| `EvoEngineer` | LLM-driven evolutionary search with structured operators |
| `FunSearch` | Program search with an island-model database |
| `EoH` | Evolution of Heuristics with crossover and mutation operators |

## Reference Task Families

These task families are included as reference adapters that validate the framework across different solution spaces.

| Task Family | Role in the package |
|---|---|
| Scientific regression | CPU-reviewable reference task for symbolic regression workflows |
| Prompt engineering | CPU-reviewable reference task for string optimization workflows |
| Adversarial attacks | CPU-reviewable reference task for algorithm evolution workflows |
| Control Box2D | CPU-reviewable reference task for policy evolution workflows |
| CUDA engineering | Optional hardware-backed reference task family; the reviewed surface focuses on the task shell, fake-mode paths, and prompt interfaces |
| CANN init | Experimental adjacent workflow for Ascend NPU operator generation; not part of the primary reviewed surface |

## Quick Start

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.tools import HttpsApi

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)

llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key-here",
    model="gpt-4o",
)

result = evotoolkit.solve(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=5,
)
```

## Tutorials And Examples

- Docs: <https://evotoolkit.readthedocs.io/>
- Tutorials walk through reference task families end to end
- `examples/` contains runnable tutorial scripts and reproducibility assets
- Hardware-backed examples remain optional and are not part of the primary reviewed surface

## Reviewer-Facing Documents

- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Prior-work software deltas: [SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md](SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md)
- Reviewed surface definition: [REVIEWED_SURFACE.md](REVIEWED_SURFACE.md)
- Testing and coverage commands: [TESTING.md](TESTING.md)

## Citation

If you use EvoToolkit in research, please cite the software release or repository snapshot you used. A repository citation entry is:

```bibtex
@software{guo2026evotoolkit,
  author = {Guo, Ping and Zhang, Qingfu},
  title = {evotoolkit},
  year = {2026},
  url = {https://github.com/pgg3/evotoolkit},
  version = {1.0.0}
}
```

## License

EvoToolkit is released under the MIT License. Optional hardware-backed workflows may depend on separately installed third-party toolchains; see the task-specific guides for those external requirements.
