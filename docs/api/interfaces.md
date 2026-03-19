# Interfaces API

Interfaces connect optimization tasks to evolutionary algorithms, handling algorithm-specific adaptations.

---

## What are Interfaces?

An **Interface** is a bridge between a **Task** (what you want to optimize) and a **Method** (how to optimize it).

```
Task (Problem) → Interface (Adapter) → Method (Algorithm)
```

Interfaces handle:
- Algorithm-specific prompt generation for LLM
- Task-specific operators (mutation, crossover, etc.)
- Solution format conversion
- Evaluation orchestration

---

## Python Task Interfaces

### Python Interfaces

See the dedicated pages:

- [EvoEngineerPythonInterface](interfaces/python/evoengineer-python-interface.md)
- [FunSearchPythonInterface](interfaces/python/funsearch-python-interface.md)
- [EoHPythonInterface](interfaces/python/eoh-python-interface.md)

---

---

---

---

---

## CUDA Task Interfaces

See the dedicated pages:

- [EvoEngineerFullCudaInterface](interfaces/cuda/evoengineer-full-cuda-interface.md)
- [EvoEngineerFreeCudaInterface](interfaces/cuda/evoengineer-free-cuda-interface.md)
- [EvoEngineerInsightCudaInterface](interfaces/cuda/evoengineer-insight-cuda-interface.md)
- [FunSearchCudaInterface](interfaces/cuda/funsearch-cuda-interface.md)
- [EoHCudaInterface](interfaces/cuda/eoh-cuda-interface.md)

See [CUDA Task Tutorial](../tutorials/built-in/cuda-task.md) for details.

---

## Base Interface Class

### BaseMethodInterface

Base class for all method interfaces. See the reference page:
[BaseMethodInterface](interfaces/base-method-interface.md).

**Key Methods:**

- `make_init_sol()`: Create initial solution
- `parse_response(response_str)`: Parse LLM response into Solution
- `get_operator_prompt(operator, population, **kwargs)`: Generate LLM prompt for a given operator

**Creating Custom Interfaces:**

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.core import Solution

class MyCustomInterface(EvoEngineerPythonInterface):
    def get_operator_prompt(self, operator, population, **kwargs):
        # Your custom prompt generation logic
        best_sol = max(
            (s for s in population if s.evaluation_res and s.evaluation_res.valid),
            key=lambda s: s.evaluation_res.score,
            default=None,
        )
        prompt = f"Improve this solution:\n{best_sol.sol_string if best_sol else 'No solution yet'}"
        return prompt
```

---

## Interface Selection Guide

### For Python Tasks

| Task Type | Recommended Interface | Alternative |
|-----------|----------------------|-------------|
| Scientific Regression | `EvoEngineerPythonInterface` | `FunSearchPythonInterface` |
| General Optimization | `EvoEngineerPythonInterface` | `EoHPythonInterface` |
| Quick Prototyping | `EoHPythonInterface` | `EvoEngineerPythonInterface` |

### For CUDA Tasks

| Task Type | Recommended Interface |
|-----------|-----------------------|
| Kernel Optimization | `EvoEngineerCudaInterface` |
| GPU Algorithm Discovery | `FunSearchCudaInterface` |

---

## How Interfaces Work

### 1. Prompt Generation

Interfaces create algorithm-specific prompts for the LLM:

```python
# EvoEngineer prompt example
prompt = """
You are evolving a Python function to approximate data.

Previous generation best solution:
{previous_best_code}

Current fitness: {fitness}

Please improve this solution or create a new one.
"""
```

### 2. Response Parsing

Interfaces extract code from LLM responses:

```python
response = llm_api.call(prompt)
solution = interface.parse_response(response)
# solution.sol_string now contains the extracted Python/CUDA code
```

### 3. Operator Prompts

Interfaces generate algorithm-specific prompts for each operator (e.g., init, mutation, crossover). The LLM receives these prompts and returns a new solution:

```python
# Internally, interfaces call get_operator_prompt() to build prompts
# and parse_response() to extract solutions from LLM output
operator_prompt = interface.get_operator_prompt(operator, population)
```

---

## Advanced: Custom Interfaces

Create a custom interface for specialized algorithms or tasks:

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.core import Solution

class MySpecializedInterface(EvoEngineerPythonInterface):
    def __init__(self, task):
        super().__init__(task)
        self.custom_config = self.load_custom_config()

    def get_operator_prompt(self, operator, population, **kwargs):
        # Custom prompt with domain-specific instructions
        valid_sols = [s for s in population if s.evaluation_res and s.evaluation_res.valid]
        best_sol = max(valid_sols, key=lambda s: s.evaluation_res.score) if valid_sols else None

        base_prompt = super().get_operator_prompt(operator, population, **kwargs)
        domain_context = f"\nDomain-specific context: {self.custom_config['context']}\n"
        return domain_context + base_prompt

    def load_custom_config(self):
        # Load domain-specific configuration
        return {"context": "Custom domain knowledge"}
```

**Usage:**

```python
task = MyCustomTask()
interface = MySpecializedInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

---

## Comparison: Interface vs Direct Method Call

### Using Interface (High-Level API) ✅ Recommended

```python
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

**Advantages:**
- Simple and concise
- Automatic configuration
- Algorithm is inferred from interface

### Direct Method Call (Low-Level API) ⚙️ Advanced

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(interface=interface, ...)
algorithm = EvoEngineer(config)
algorithm.run()
```

**Advantages:**
- Full control over configuration
- Access to internal state
- Custom post-processing

---

## Next Steps

- See [Tasks API](tasks.md) for available optimization tasks
- Check [Methods API](methods.md) for evolutionary algorithms
- Try the [Advanced Usage Tutorial](../tutorials/advanced-overview.md) for low-level API

