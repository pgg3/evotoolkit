# EvoToolkit

**LLM-driven evolutionary optimization for executable solutions**

EvoToolkit is a Python framework for evolving code, symbolic expressions, prompts, and other executable text with large language models. It exposes a reusable `Method -> Interface -> Task` architecture so that algorithms and application domains can be composed instead of rebuilt from scratch.

---

## Key Features

- **LLM-driven evolution** for generating and refining candidate solutions
- **Multiple algorithms**: EoH, EvoEngineer, and FunSearch
- **Reusable architecture** across Python, string, CUDA, control, and CANN tasks
- **Simple top-level API** via `evotoolkit.solve(...)`
- **Bilingual documentation** with tutorials and API reference

### Built-in Task Families

| Task Family | Description | Details |
|-------------|-------------|---------|
| Scientific regression | Symbolic regression on curated datasets | [Scientific Regression Tutorial](tutorials/built-in/scientific-regression.md) |
| Prompt engineering | Optimize prompt templates for downstream tasks | [Prompt Engineering Tutorial](tutorials/built-in/prompt-engineering.md) |
| Adversarial attacks | Evolve proposal-generation algorithms | [Adversarial Attack Tutorial](tutorials/built-in/adversarial-attack.md) |
| CUDA engineering | Evolve and optimize CUDA kernels | [CUDA Task Tutorial](tutorials/built-in/cuda-task.md) |
| Control (Box2D) | Evolve interpretable control policies | [Control Box2D Tutorial](tutorials/built-in/control-box2d.md) |
| CANN init | Generate Ascend C operator kernels | [CANN Init Tutorial](tutorials/built-in/cann-init.md) |

---

## Quick Start

### Installation

```bash
pip install evotoolkit

# Optional extras
pip install "evotoolkit[scientific_regression]"
pip install "evotoolkit[prompt_engineering]"
pip install "evotoolkit[adversarial_attack]"
pip install "evotoolkit[cuda_engineering]"
pip install "evotoolkit[control_box2d]"
pip install "evotoolkit[cann_init]"
pip install "evotoolkit[all_tasks]"
```

The public package is tested on Python 3.10-3.12. Optional CUDA and CANN workflows additionally require the corresponding hardware/toolchains.

### Your First Optimization

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

For a full walkthrough, see [Getting Started](getting-started.md).

---

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **EvoEngineer** | LLM-driven evolutionary search with structured operators |
| **FunSearch** | Program search with an island-model database |
| **EoH** | Evolution of Heuristics with crossover and mutation operators |

---

## Documentation

- **[Installation](installation.md)** for environment setup
- **[Getting Started](getting-started.md)** for the first runnable workflow
- **[Tutorials](tutorials/index.md)** for end-to-end examples
- **[API Reference](api/index.md)** for public interfaces and tasks
- **[Development](development/contributing.md)** for contributor workflows

---

## Links

- **GitHub**: [https://github.com/pgg3/evotoolkit](https://github.com/pgg3/evotoolkit)
- **PyPI**: [https://pypi.org/project/evotoolkit/](https://pypi.org/project/evotoolkit/)
- **Changelog**: [CHANGELOG.md](https://github.com/pgg3/evotoolkit/blob/master/CHANGELOG.md)
- **Prior-work improvements**: [SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md](https://github.com/pgg3/evotoolkit/blob/master/SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md)

---

## License

EvoToolkit is released under the MIT License. Optional hardware-oriented workflows may depend on separately installed third-party toolchains; see each task guide for those external requirements.

---

## Citation

If you use EvoToolkit in research, please cite the software version or repository snapshot you used. A repository citation entry is:

```bibtex
@software{guo2026evotoolkit,
  author = {Guo, Ping and Zhang, Qingfu},
  title = {evotoolkit},
  year = {2026},
  url = {https://github.com/pgg3/evotoolkit},
  version = {1.0.0rc6}
}
```

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/pgg3/evotoolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions)
- **Email**: pguo6680@gmail.com
