# EvoToolkit

**LLM-driven solution evolutionary optimization toolkit**

EvoToolkit is a Python library that leverages Large Language Models (LLMs) to evolve solutions for optimization problems. It combines the power of evolutionary algorithms with LLM-based solution generation and refinement, supporting code, text, and other evaluable representations.

---

## ✨ Key Features

- **🤖 LLM-Driven Evolution**: Use state-of-the-art language models to generate and evolve solutions
- **🔬 Multiple Algorithms**: Support for EoH, EvoEngineer, and FunSearch evolutionary methods
- **🌍 Task-Agnostic**: Supports any evaluable optimization task (code, text, math expressions, etc.)
- **🎯 Extensible Framework**: Easy-to-extend task system for custom optimization problems
- **🔌 Simple API**: High-level `evotoolkit.solve()` function for quick prototyping
- **🛠️ Advanced Customization**: Low-level API for fine-grained control

### Built-in Task Types

| Task Type | Description | Details |
|-----------|-------------|---------|
| **🔬 Scientific Regression** | Symbolic regression on real scientific datasets | [Scientific Regression Tutorial](tutorials/built-in/scientific-regression.md) |
| **💬 Prompt Engineering** | Optimize LLM prompts for downstream tasks | [Prompt Engineering Tutorial](tutorials/built-in/prompt-engineering.md) |
| **🛡️ Adversarial Attacks** | Evolve adversarial attack algorithms | [Adversarial Attack Tutorial](tutorials/built-in/adversarial-attack.md) |
| **⚡ CUDA Code Evolution** | Evolve and optimize CUDA kernels | [CUDA Task Tutorial](tutorials/built-in/cuda-task.md) |

---

## 🚀 Quick Start

### Installation

```bash
pip install evotoolkit

# Or install with all dependencies
pip install evotoolkit[all]
```

For detailed installation instructions, see the [Installation Guide](installation.md).

### Your First Optimization

```python
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

# 1. Create a task
task = ScientificRegressionTask(dataset_name="bactgrow")

# 2. Create an interface
interface = EvoEngineerPythonInterface(task)

# 3. Solve with LLM
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key-here",
    model="gpt-4o"
)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

That's it! EvoToolkit will use the LLM to evolve mathematical equations to fit your scientific data.

For a complete walkthrough, check out the [Getting Started Guide](getting-started.md).

---

## 📚 Available Algorithms

| Algorithm | Description |
|-----------|-------------|
| **EvoEngineer** | Main LLM-driven evolutionary algorithm |
| **FunSearch** | Function search optimization method |
| **EoH** | Evolution of Heuristics |

See the [Tutorials](tutorials/index.md) section for more usage examples.

---

## 📖 Documentation

- **[Installation](installation.md)**: Installation instructions and setup
- **[Getting Started](getting-started.md)**: Quick start guide and basic usage
- **[Tutorials](tutorials/index.md)**: Step-by-step tutorials for common tasks
- **[API Reference](api/index.md)**: Detailed API documentation
- **[Development](development/contributing.md)**: Contributing guidelines and architecture

---

## 🔗 Links

- **GitHub**: [https://github.com/pgg3/evotoolkit](https://github.com/pgg3/evotoolkit)
- **PyPI**: [https://pypi.org/project/evotoolkit/](https://pypi.org/project/evotoolkit/)
- **Paper**: arXiv (submitted)

---

## 📄 License

EvoToolkit is dual-licensed:

- **Academic & Open Source Use**: Free for academic research, education, and open source projects. **Citation required** for academic publications.
- **Commercial Use**: Requires a separate commercial license. Contact pguo6680@gmail.com for licensing.

See [LICENSE](https://github.com/pgg3/evotoolkit/blob/master/LICENSE) for full terms.

---

## 🙏 Citation

If you use EvoToolkit in your research, please cite:

```bibtex
@article{guo2025evotoolkit,
  title={evotoolkit: A Unified LLM-Driven Evolutionary Framework for Generalized Solution Search},
  author={Guo, Ping and Zhang, Qingfu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  note={Submitted to arXiv}
}
```

---

## 💬 Getting Help

- **Issues**: [GitHub Issues](https://github.com/pgg3/evotoolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions)
- **Email**: pguo6680@gmail.com
