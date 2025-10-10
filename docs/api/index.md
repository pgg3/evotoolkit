# API Reference

Welcome to the EvoToolkit API reference documentation. This section provides detailed information about all public APIs, classes, and functions.

---

## Overview

EvoToolkit is organized into several main modules:

- **[Core API](core.md)**: Core functionality including `evotoolkit.solve()`, `Solution`, `Task`, and base classes
- **[Tasks](tasks.md)**: Built-in optimization tasks (Python and CUDA)
- **[Methods](methods.md)**: Evolutionary algorithms (EoH, EvoEngineer, FunSearch)
- **[Interfaces](interfaces.md)**: Method interfaces that connect tasks to algorithms
- **[Tools](tools.md)**: Utilities and LLM API clients

---

## Quick API Reference

### High-Level API

The simplest way to use EvoToolkit:

```python
import evotoolkit

result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

See [Core API: evotoolkit.solve()](core/solve.md) for details.

### Core Classes

| Class | Description | Documentation |
|-------|-------------|---------------|
| `Solution` | Represents a candidate solution | [Core API](core/solution.md) |
| `Task` | Base class for optimization tasks | [Core API](core/base-task.md) |
| `MethodInterface` | Base class for algorithm interfaces | [Interfaces](interfaces.md) |

### Built-in Tasks

| Task | Description | Documentation |
|------|-------------|---------------|
| `ScientificRegressionTask` | Scientific symbolic regression task | [Tasks](tasks.md#scientificregressiontask) |
| `PythonTask` | Generic Python task | [Tasks](tasks.md#pythontask) |
| `CudaTask` | GPU kernel optimization task | [Tasks](tasks.md#cudatask) |

### Evolutionary Algorithms

| Algorithm | Description | Documentation |
|-----------|-------------|---------------|
| `EvoEngineer` | Main LLM-driven evolutionary algorithm | [Methods](methods/evoengineer.md) |
| `FunSearch` | Function search optimization | [Methods](methods/funsearch.md) |
| `EoH` | Evolution of Heuristics | [Methods](methods/eoh.md) |

---

## API Design Philosophy

EvoToolkit provides two levels of API:

### 1. High-Level API (Recommended)

The high-level API through `evotoolkit.solve()` handles most complexity automatically:

```python
# Create task and interface
task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)

# Solve
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

**Advantages**:
- Simple and concise
- Automatic configuration
- Best for most use cases

### 2. Low-Level API (Advanced)

The low-level API provides fine-grained control:

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

# Create custom configuration
config = EvoEngineerConfig(
    task=task,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    pop_size=10,
    # ... more custom settings
)

# Create and run algorithm
algorithm = EvoEngineer(config)
algorithm.run()

# Get best solution
best_solution = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)
```

**Advantages**:
- Full customization
- Access to internal state
- Advanced debugging

See [Advanced Usage Tutorial](../tutorials/advanced-overview.md) for details.

---

## Module Organization

```
evotool/
├── __init__.py              # High-level API (solve function)
├── core/                    # Core abstractions
│   ├── base_task.py        # Task base class
│   ├── solution.py         # Solution class
│   ├── base_method.py      # Algorithm base class
│   ├── base_config.py      # Configuration base class
│   └── method_interface/   # Algorithm interfaces
├── evo_method/             # Evolutionary algorithms
│   ├── eoh/               # EoH implementation
│   ├── evoengineer/       # EvoEngineer implementation
│   └── funsearch/         # FunSearch implementation
├── task/                   # Task implementations
│   ├── python_task/       # Python task framework
│   ├── cuda_engineering/  # CUDA task framework
│   └── string_optimization/ # String optimization tasks
├── tools/                  # Utilities
│   └── llm.py             # LLM API client (HttpsApi)
└── data/                   # Data management utilities
```

---

## Common Patterns

### Pattern 1: Basic Optimization

```python
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface, './results', llm_api, max_generations=5)
```

### Pattern 2: Custom Task

```python
from evotoolkit.core import BaseTask, Solution

class MyTask(BaseTask):
    def evaluate(self, solution: Solution) -> float:
        # Your evaluation logic
        return fitness_value

task = MyTask()
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface, './results', llm_api)
```

### Pattern 3: Algorithm Comparison

```python
algorithms = [
    ('EoH', EoHPythonInterface(task)),
    ('EvoEngineer', EvoEngineerPythonInterface(task)),
    ('FunSearch', FunSearchPythonInterface(task))
]

for name, interface in algorithms:
    result = evotoolkit.solve(interface, f'./results/{name}', llm_api)
    print(f"{name}: {result.fitness}")
```

---

## API Versioning

EvoToolkit follows [Semantic Versioning](https://semver.org/):

- **Major version** (1.x.x): Breaking API changes
- **Minor version** (x.1.x): New features, backward compatible
- **Patch version** (x.x.1): Bug fixes, backward compatible

Check the current version:

```python
import evotoolkit
print(evotoolkit.__version__)  # e.g., "1.0.0"
```

---

## Type Hints

EvoToolkit uses type hints throughout the codebase. Use a type checker like `mypy` for static analysis:

```bash
pip install mypy
mypy your_script.py
```

---

## Next Steps

- Browse the [Core API](core.md) documentation
- Explore [Tasks API](tasks.md) for built-in tasks
- Check [Methods API](methods.md) for evolutionary algorithms
- Learn about [Interfaces API](interfaces.md) for algorithm integration
