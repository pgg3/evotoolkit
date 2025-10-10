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

- `generate_prompt(generation, population)`: Creates LLM prompts
- `parse_llm_response(response)`: Parses LLM output into solutions
- `mutate(solution)`: Applies mutation operator
- `crossover(parent1, parent2)`: Applies crossover operator

**Creating Custom Interfaces:**

```python
from evotoolkit.core.method_interface import BaseMethodInterface
from evotoolkit.core import Solution

class MyCustomInterface(BaseMethodInterface):
    def generate_prompt(self, generation, population):
        # Your custom prompt generation
        return prompt_string

    def parse_llm_response(self, response):
        # Parse LLM response
        code = self.extract_code(response)
        return Solution(code=code)

    def mutate(self, solution):
        # Your custom mutation logic
        return mutated_solution
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
solution = interface.parse_llm_response(response)
# solution.sol_string now contains the extracted Python/CUDA code
```

### 3. Operator Application

Interfaces apply evolutionary operators:

```python
# Mutation
mutated = interface.mutate(solution)

# Crossover
offspring = interface.crossover(parent1, parent2)
```

---

## Advanced: Custom Interfaces

Create a custom interface for specialized algorithms or tasks:

```python
from evotoolkit.core.method_interface import BaseMethodInterface
from evotoolkit.core import Solution

class MySpecializedInterface(BaseMethodInterface):
    def __init__(self, task):
        super().__init__(task)
        self.custom_config = self.load_custom_config()

    def generate_prompt(self, generation, population):
        # Custom prompt with domain-specific instructions
        best_sol = max(population, key=lambda s: s.evaluation_res.score if s.evaluation_res.valid else float('-inf'))

        prompt = f"""
        Domain-specific context: {self.custom_config['context']}

        Evolve a solution that improves upon:
        {best_sol.sol_string}

        Current best score: {best_sol.evaluation_res.score}
        Generation: {generation}
        """
        return prompt

    def parse_llm_response(self, response):
        # Custom parsing logic
        code = self.extract_code_with_custom_markers(response)
        return Solution(code=code)

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

