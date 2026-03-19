# EvoToolkit

[![CI](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pgg3/evotoolkit/branch/master/graph/badge.svg)](https://codecov.io/gh/pgg3/evotoolkit)
[![PyPI](https://img.shields.io/pypi/v/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![Python](https://img.shields.io/pypi/pyversions/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LLM-driven evolutionary optimization for executable solutions**

EvoToolkit is a Python framework for evolving code, symbolic expressions, prompts, and other executable text with large language models. It provides a unified `Method -> Interface -> Task` architecture across EoH, EvoEngineer, and FunSearch.

## Installation

Core installation:

```bash
pip install evotoolkit
```

Optional task extras:

```bash
pip install "evotoolkit[scientific_regression]"
pip install "evotoolkit[prompt_engineering]"
pip install "evotoolkit[adversarial_attack]"
pip install "evotoolkit[cuda_engineering]"
pip install "evotoolkit[control_box2d]"
pip install "evotoolkit[cann_init]"
pip install "evotoolkit[all_tasks]"
```

The published package is tested on Python 3.10-3.12. Optional CUDA and CANN workflows require the corresponding hardware/toolchains in addition to the Python extras above.

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

## Project Facts

- Three built-in evolutionary algorithms: EoH, EvoEngineer, FunSearch
- A unified top-level API: `evotoolkit.solve(...)`
- Bilingual documentation with tutorials and API reference
- MIT-licensed core framework with optional hardware-specific extensions
- PyPI distribution and GitHub-hosted source/documentation

## Documentation

- Docs: <https://pgg3.github.io/evotoolkit/>
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Prior-work software deltas: [SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md](SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md)
- Testing and coverage commands: [TESTING.md](TESTING.md)

## Citation

If you use EvoToolkit in research, please cite the software release or repository snapshot you used. A repository citation entry is:

```bibtex
@software{guo2026evotoolkit,
  author = {Guo, Ping and Zhang, Qingfu},
  title = {evotoolkit},
  year = {2026},
  url = {https://github.com/pgg3/evotoolkit},
  version = {1.0.0rc6}
}
```

## License

EvoToolkit is released under the MIT License. Hardware-specific workflows may depend on separately installed third-party toolchains; see the task-specific installation guides for those requirements.
