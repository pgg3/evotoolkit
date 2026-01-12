# Customizing Evolution Methods

Learn how to customize evolutionary behavior in EvoToolkit by modifying prompts or developing entirely new algorithms.

---

## Overview

The quality of evolutionary optimization in EvoToolkit is controlled by:

1. **Evolution Method**: The algorithm framework (EvoEngineer, EoH, FunSearch)
2. **Interface**: The bridge between tasks and methods, containing prompt logic
3. **Prompts**: Instructions sent to the LLM to guide solution generation

This tutorial covers two levels of customization:

- **Level 1: Customize prompts** - Inherit existing Interfaces and modify prompts (recommended)
- **Level 2: Develop new algorithms** - Create entirely new evolutionary strategies (advanced)

---

## Level 1: Customizing Prompts

### 1.1 Understanding Interfaces

Each evolution method uses an **Interface** class that:

- Defines operators (init, mutation, crossover, etc.)
- Generates LLM prompts for each operator via `get_operator_prompt()`
- Parses LLM responses into solutions

**Available Interfaces:**

| Interface | Method | Description |
|-----------|--------|-------------|
| `EvoEngineerPythonInterface` | EvoEngineer | Main LLM-driven algorithm for Python tasks |
| `EoHPythonInterface` | EoH | Evolution of Heuristics for Python tasks |
| `FunSearchPythonInterface` | FunSearch | Function search for Python tasks |
| `EvoEngineerCUDAInterface` | EvoEngineer | For CUDA code evolution |

### 1.2 Inspecting Existing Prompts

Before customizing, examine how existing Interfaces generate prompts:

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
import inspect

# Create an interface
interface = EvoEngineerPythonInterface(task)

# View the source code of the prompt generation method
print(inspect.getsource(interface.get_operator_prompt))
```

This shows you:

- What information is included in prompts
- How prompts are structured
- What format LLMs are expected to follow

### 1.3 Creating Custom Interfaces

To customize prompts, inherit from an existing Interface and override `get_operator_prompt()`:

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.core import Solution
from typing import List

class CustomInterface(EvoEngineerPythonInterface):
    """Custom Interface with modified prompts"""

    def get_operator_prompt(self, operator_name: str,
                           selected_individuals: List[Solution],
                           current_best_sol: Solution,
                           random_thoughts: List[str],
                           **kwargs) -> List[dict]:
        """Override this method to customize prompts for any operator"""

        # Get base task description
        task_description = self.task.get_base_task_description()

        if operator_name == "mutation":
            # Custom mutation prompt
            prompt = f"""You are an expert optimizer.
Current best solution score: {current_best_sol.evaluation_res.score:.5f}

Your task: {task_description}

Current code:
{current_best_sol.sol_string}

Generate an improved solution by applying a mutation.
Focus on: [YOUR CUSTOM REQUIREMENTS HERE]

Format:
- name: descriptive_name
- code: [complete code]
- thought: [reasoning]
"""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "crossover":
            # Custom crossover prompt
            parent1, parent2 = selected_individuals[0], selected_individuals[1]
            prompt = f"""Combine these two solutions...
Parent 1 (score {parent1.evaluation_res.score:.5f}):
{parent1.sol_string}

Parent 2 (score {parent2.evaluation_res.score:.5f}):
{parent2.sol_string}

Create an offspring combining their strengths...
"""
            return [{"role": "user", "content": prompt}]

        # Use default implementation for other operators
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# Use your custom Interface
custom_interface = CustomInterface(task)
result = evotoolkit.solve(
    interface=custom_interface,
    output_path='./custom_results',
    running_llm=llm_api,
    max_generations=10
)
```

### 1.4 Prompt Engineering Best Practices

When customizing prompts:

#### 1.4.1 Be Specific About Requirements

```python
# Vague
prompt = "Improve this code"

# Specific
prompt = """Improve this code by:
1. Reducing computational complexity
2. Maintaining numerical stability
3. Ensuring correctness on edge cases"""
```

#### 1.4.2 Provide Context and Examples

```python
prompt = f"""Task: {task_description}

Good practices:
- Use vectorized NumPy operations
- Avoid loops when possible
- Handle edge cases (empty arrays, zero values)

Bad practices:
- Explicit Python loops over large arrays
- Division without checking for zeros

Current code:
{current_best_sol.sol_string}

Generate an improved version..."""
```

#### 1.4.3 Include Domain Knowledge

```python
# For scientific regression
prompt = """Base equations on known physical/biological principles:
- Monod equation for substrate limitation: μ = μmax * S / (Ks + S)
- Arrhenius equation for temperature: k = A * exp(-Ea / RT)
- Logistic growth for population dynamics
..."""

# For CUDA optimization
prompt = """Apply GPU optimization techniques:
- Coalesced memory access
- Shared memory for frequently accessed data
- Minimize divergent branches
..."""
```

#### 1.4.4 Customize by Operator Type

Different operators benefit from different prompts:

```python
def get_operator_prompt(self, operator_name, ...):
    if operator_name == "init":
        # Initial exploration - encourage diversity
        prompt = "Explore diverse solution approaches..."

    elif operator_name == "mutation":
        # Local search - small improvements
        prompt = "Make incremental improvements to current solution..."

    elif operator_name == "crossover":
        # Combine features - recombination
        prompt = "Combine strengths from both parent solutions..."
```

---

## Level 2: Developing New Algorithms

!!! warning "Advanced Topic"
    This section is for users who want to implement entirely new evolutionary strategies. Most users should start with Level 1 (customizing prompts) which is often sufficient.

### 2.1 When to Develop New Algorithms

Consider developing a new algorithm when:

- Existing algorithms (EvoEngineer, EoH, FunSearch) don't fit your problem structure
- You have domain-specific evolutionary strategies
- You want to research novel LLM-driven optimization approaches
- You need completely different evolutionary workflows or selection mechanisms

### 2.2 Algorithm Architecture

EvoToolkit uses a three-layer architecture to implement new algorithms:

```
┌─────────────────────────────────────────┐
│  Layer 1: Algorithm Class               │
│  - Inherits from Method base class      │
│  - Implements run() method (main loop)  │
│  - Defines Config class (parameters)    │
│  Location: evo_method/your_algorithm/   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Layer 2: Generic Interface Base Class  │
│  - Only requires: parse_response()      │
│  - Other methods: algorithm-specific    │
│  Location: core/method_interface/       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Layer 3: Task-Specific Interface       │
│  - Inherits from generic Interface      │
│  - Implements task-specific logic       │
│  Location: task/*/method_interface/     │
└─────────────────────────────────────────┘
```

!!! important "Interface Design Flexibility"
    **Core Requirements:** `BaseMethodInterface` only mandates one method:

    - `parse_response(response_str)` - Parse LLM response

    **Algorithm-Specific Methods:** All other methods are defined by your algorithm's needs:

    - **Operator-based** (like EvoEngineer): `get_init_operators()`, `get_offspring_operators()`, `get_operator_prompt()`
    - **Iterative** (like FunSearch): `generate_evolution_prompt()`
    - **Your design**: Define any methods your algorithm requires

**Existing Algorithm Examples:**

- **EvoEngineer**: `evo_method/evoengineer/evoengineer.py` (Layer 1) → `core/method_interface/evoengineer_interface.py` (Layer 2) → `task/python_task/method_interface/evoengineer_interface.py` (Layer 3)
- **EoH**: `evo_method/eoh/` → `core/method_interface/eoh_interface.py` → `task/python_task/method_interface/eoh_interface.py`
- **FunSearch**: `evo_method/funsearch/` → `core/method_interface/funsearch_interface.py` → `task/python_task/method_interface/funsearch_interface.py`

### 2.3 Creating a New Algorithm

!!! note "Complete Implementation Required"
    Creating a new algorithm requires implementing all three layers. For most customization needs, Level 1 (custom prompts) is sufficient.

Due to the complexity of implementing a full three-layer architecture, we recommend:

1. **Study existing implementations** - See section 2.4 below
2. **Start with prompt customization** (Level 1) - Much easier and often sufficient
3. **Extend existing algorithms** - Inherit from EvoEngineerInterface rather than creating from scratch

For a complete implementation guide, refer to the existing algorithm implementations in the source code

### 2.4 Example: Temperature-based Mutation Algorithm

Here's a complete example of a custom algorithm with temperature-controlled mutation:

```python
from evotoolkit.core import EvoEngineerInterface, Operator, Solution
from evotoolkit.task.python_task import PythonTask
from typing import List
import math

class TemperatureBasedEvolution(EvoEngineerInterface):
    """Custom algorithm with simulated annealing-style temperature"""

    def __init__(self, task: PythonTask, initial_temp=10.0, cooling_rate=0.9):
        super().__init__(task)
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.generation = 0

    def get_init_operators(self):
        return [Operator("init", 0)]

    def get_offspring_operators(self):
        return [
            Operator("hot_mutation", 1),   # Large changes
            Operator("cool_mutation", 1),  # Small changes
            Operator("crossover", 2),
        ]

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):

        task_description = self.task.get_base_task_description()

        # Cool down temperature each generation
        self.temperature *= self.cooling_rate
        self.generation += 1

        if operator_name == "init":
            prompt = f"""Initialize solution for: {task_description}"""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "hot_mutation":
            # High temperature = large changes
            if self.temperature > 5.0:
                prompt = f"""Make a BOLD, exploratory change to:
{selected_individuals[0].sol_string}

Try a completely different approach or algorithm.
"""
            else:
                # Fallback to regular mutation at low temp
                prompt = f"""Make a moderate change to:
{selected_individuals[0].sol_string}
"""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "cool_mutation":
            # Low temperature = small refinements
            prompt = f"""Make a SMALL, refinement change to:
{selected_individuals[0].sol_string}

Focus on minor improvements: better constants, edge cases, small optimizations.
"""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "crossover":
            parent1, parent2 = selected_individuals[0], selected_individuals[1]
            prompt = f"""Combine these solutions:
Parent 1 (score {parent1.evaluation_res.score}):
{parent1.sol_string}

Parent 2 (score {parent2.evaluation_res.score}):
{parent2.sol_string}
"""
            return [{"role": "user", "content": prompt}]

# Use the custom algorithm
task = MyTask(...)
algo = TemperatureBasedEvolution(task, initial_temp=10.0, cooling_rate=0.9)
result = evotoolkit.solve(interface=algo, running_llm=llm_api, max_generations=20)
```

---

## Task-Specific Customization Examples

### 3.1 For Scientific Regression

```python
class ScientificInterface(EvoEngineerPythonInterface):
    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        if operator_name == "mutation":
            prompt = f"""You are a physicist/biologist discovering equations.

Current equation (MSE: {current_best_sol.evaluation_res.score:.6f}):
{current_best_sol.sol_string}

Use established principles:
- Monod: μ = μmax * S / (Ks + S)
- Arrhenius: k = A * exp(-Ea / RT)
- Michaelis-Menten kinetics
- Logistic growth

Constraints:
- Ensure dimensional consistency
- Avoid numerical instabilities
- Keep model parsimonious

Generate improved equation..."""
            return [{"role": "user", "content": prompt}]
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)
```

### 3.2 For CUDA Optimization

```python
class CUDAInterface(EvoEngineerCUDAInterface):
    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        if operator_name == "mutation":
            prompt = f"""You are a GPU optimization expert.

Current CUDA kernel (time: {current_best_sol.evaluation_res.score:.3f}ms):
{current_best_sol.sol_string}

Apply optimizations:
- Coalesced memory access patterns
- Shared memory for temporary data
- Reduce bank conflicts
- Minimize thread divergence
- Optimize block/grid dimensions

Generate optimized kernel..."""
            return [{"role": "user", "content": prompt}]
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)
```

### 3.3 For Prompt Engineering

```python
class PromptOptimizationInterface(EvoEngineerPythonInterface):
    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        if operator_name == "mutation":
            prompt = f"""You are an expert in LLM prompt engineering.

Current prompt (score: {current_best_sol.evaluation_res.score:.3f}):
{current_best_sol.sol_string}

Improvement strategies:
- Add clear instructions and structure
- Provide relevant examples
- Specify output format
- Include constraints and guidelines
- Use appropriate tone and style

Generate improved prompt..."""
            return [{"role": "user", "content": prompt}]
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)
```

---

## Testing and Debugging

### 4.1 Logging Prompts

To see what prompts are sent to the LLM:

```python
class DebugInterface(EvoEngineerPythonInterface):
    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):

        prompts = super().get_operator_prompt(operator_name, selected_individuals,
                                             current_best_sol, random_thoughts, **kwargs)

        # Log prompts for debugging
        print(f"\n{'='*60}")
        print(f"OPERATOR: {operator_name}")
        print(f"PROMPT:\n{prompts[0]['content']}")
        print(f"{'='*60}\n")

        return prompts
```

### 4.2 Validating Custom Interfaces

Before running full evolution, test your Interface:

```python
# Create interface
interface = CustomInterface(task)

# Get initial solution
init_sol = task.make_init_sol_wo_other_info()

# Test prompt generation for each operator
for op in interface.get_offspring_operators():
    prompts = interface.get_operator_prompt(
        operator_name=op.name,
        selected_individuals=[init_sol],
        current_best_sol=init_sol,
        random_thoughts=[]
    )
    print(f"Operator {op.name}:")
    print(prompts[0]['content'][:200] + "...")
    print()
```

---

## Next Steps

- **Experiment**: Try different prompt styles and see what works best
- **Analyze**: Compare results across different customizations
- **Share**: Consider contributing successful customizations to the project

**Related Documentation:**

- [Scientific Regression Tutorial](../built-in/scientific-regression.md) - Example application
- [CUDA Task Tutorial](../built-in/cuda-task.md) - GPU code optimization
- [Advanced Usage](../advanced-overview.md) - More configuration options
- [API Reference](../../api/index.md) - Complete Interface API docs
- [Contributing](../../development/contributing.md) - Share your custom methods
