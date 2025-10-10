# Tutorials

Welcome to the EvoToolkit tutorials! These step-by-step guides will help you master evolutionary optimization with LLMs.

---

## Getting Started

New to EvoToolkit? Start here:

1. **[Installation](../installation.md)** - Set up your environment
2. **[Getting Started](../getting-started.md)** - Run your first optimization in 5 minutes
3. **[Scientific Regression Tutorial](built-in/scientific-regression.md)** - Deep dive into a complete example

---

## Tutorial Categories

### Built-in Tasks

Learn how to use EvoToolkit's pre-built optimization tasks:

- **[Scientific Regression](built-in/scientific-regression.md)** - Discover mathematical equations from data
- **[Prompt Engineering](built-in/prompt-engineering.md)** - Optimize LLM prompts for better performance
- **[Adversarial Attack](built-in/adversarial-attack.md)** - Generate adversarial examples
- **[CUDA Tasks](built-in/cuda-task.md)** - Optimize GPU kernels for performance

### Customization

Extend EvoToolkit for your specific needs:

- **[Custom Tasks](customization/custom-task.md)** - Create your own optimization problems
- **[Customizing Evolution](customization/customizing-evolution.md)** - Modify prompts and algorithms

### Advanced

Master the low-level APIs:

- **[Advanced Usage](advanced-overview.md)** - Fine-grained control and debugging

---

## Tutorial Overview

| Tutorial | Level | Time | Topics Covered |
|----------|-------|------|----------------|
| [Scientific Regression](built-in/scientific-regression.md) | Beginner | 20 min | High-level API, real datasets, equation evolution |
| [Prompt Engineering](built-in/prompt-engineering.md) | Beginner-Intermediate | 20 min | LLM prompt optimization, task performance |
| [Adversarial Attack](built-in/adversarial-attack.md) | Intermediate | 25 min | Evolving adversarial examples, attack algorithms |
| [CUDA Tasks](built-in/cuda-task.md) | Advanced | 30 min | GPU optimization, CUDA kernels, performance |
| [Custom Tasks](customization/custom-task.md) | Intermediate | 20 min | Creating tasks, evaluation, custom fitness |
| [Customizing Evolution](customization/customizing-evolution.md) | Intermediate-Advanced | 30 min | Prompt engineering, custom algorithms, Interface development |
| [Advanced Usage](advanced-overview.md) | Advanced | 25 min | Low-level API, custom configs, debugging |

---

## Quick Reference

### Common Workflow Patterns

#### Pattern 1: Basic Optimization
```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface

interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface, './results', llm_api)
```

#### Pattern 2: Algorithm Comparison
```python
algorithms = [
    ('EoH', EoHPythonInterface(task)),
    ('EvoEngineer', EvoEngineerPythonInterface(task)),
    ('FunSearch', FunSearchPythonInterface(task))
]

for name, interface in algorithms:
    result = evotoolkit.solve(interface, f'./results/{name}', llm_api)
```

#### Pattern 3: Custom Configuration
```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=20,
    pop_size=10
)

algorithm = EvoEngineer(config)
algorithm.run()
```

---

## Downloadable Examples

All tutorial code is available as standalone Python scripts in the `examples/` directory:

- `examples/scientific_regression/` - Scientific equation discovery
- `examples/custom_task/my_custom_task.py` - Custom task implementation
- `examples/cuda_task/kernel_optimization.py` - CUDA kernel optimization
- `examples/advanced/low_level_api.py` - Low-level API usage

Clone the repository to get started:

```bash
git clone https://github.com/pgg3/evotoolkitkit.git
cd evotool/examples
```

---

## Need Help?

### Documentation & Resources

- **[API Reference](../api/index.md)** - Detailed API documentation
- **[Development Guide](../development/contributing.md)** - Contributing guidelines
- **[Advanced Examples](https://github.com/pgg3/evotoolkit/tree/master/examples)** - Complex use case references

### Community Support

- **[GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions)** - Ask questions and share projects
- **[GitHub Issues](https://github.com/pgg3/evotoolkit/issues)** - Report bugs and request features
- **[Example Gallery](https://github.com/pgg3/evotoolkit/wiki/Examples)** - Community-contributed examples
- **[Blog](https://github.com/pgg3/evotoolkit/wiki/Blog)** - Articles and case studies

### Direct Contact

- **Email**: pguo6680@gmail.com

---

## Video Tutorials

Coming soon! Subscribe to our [YouTube channel](#) for video tutorials.
