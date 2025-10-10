# Scientific Symbolic Regression Tutorial

Learn how to discover mathematical equations from real scientific datasets using LLM-driven evolution.

!!! note "Academic Citation"
    The scientific regression task and datasets are based on research from CoEvo. If you use this feature in academic work, please cite:

    ```bibtex
    @misc{guo2024coevocontinualevolutionsymbolic,
        title={CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models},
        author={Ping Guo and Qingfu Zhang and Xi Lin},
        year={2024},
        eprint={2412.18890},
        archivePrefix={arXiv},
        primaryClass={cs.AI},
        url={https://arxiv.org/abs/2412.18890}
    }
    ```

!!! tip "Complete Example Code"
    This tutorial provides complete, runnable examples (click to view/download):

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/scientific_regression/basic_example.py) - Basic usage
    - [:material-download: custom_prompt.py](https://github.com/pgg3/evotoolkit/blob/master/examples/scientific_regression/custom_prompt.py) - Custom prompt example
    - [:material-download: compare_algorithms.py](https://github.com/pgg3/evotoolkit/blob/master/examples/scientific_regression/compare_algorithms.py) - Algorithm comparison
    - [:material-file-document: README.md](https://github.com/pgg3/evotoolkit/blob/master/examples/scientific_regression/README.md) - Examples documentation and usage guide

    Run locally:
    ```bash
    cd examples/scientific_regression
    python basic_example.py
    ```

---

## Overview

This tutorial demonstrates:

- Loading scientific datasets for symbolic regression
- Discovering mathematical equations from data
- Optimizing equation parameters automatically
- Evolving complex scientific models

---

## Installation

Install scientific regression dependencies:

```bash
pip install evotoolkit[scientific_regression]
```

This installs:

- SciPy (for parameter optimization)
- Pandas (for data loading)

**Prerequisites:**

- Basic understanding of symbolic regression concepts
- Familiarity with NumPy and SciPy usage

---

## Prepare Datasets

EvoToolkit supports **lazy downloading** - datasets are automatically downloaded on first use to a default location.

**Available datasets:**

- **bactgrow**: E. Coli bacterial growth rate prediction (4 inputs: population, substrate, temp, pH)
- **oscillator1**: Damped nonlinear oscillator acceleration (2 inputs: position, velocity)
- **oscillator2**: Damped nonlinear oscillator variant 2 (2 inputs: position, velocity)
- **stressstrain**: Aluminium stress prediction (2 inputs: strain, temperature)

**Custom data directory:**

```python
# Specify data directory in task (auto-downloads on first run)
task = ScientificRegressionTask(
    dataset_name="bactgrow",
    data_dir='./my_data'
)
```

---

## Example: Bacterial Growth Modeling

### Step 1: Create the Task

```python
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask

# Create task for bacterial growth dataset
task = ScientificRegressionTask(
    dataset_name="bactgrow",
    max_params=10,          # Number of optimizable parameters
    timeout_seconds=60.0    # Timeout per evaluation
)

print(f"Dataset: {task.dataset_name}")
print(f"Train size: {task.task_info['train_size']}")
print(f"Test size: {task.task_info['test_size']}")
```

**Output:**
```
Dataset: bactgrow
Train size: 7500
Test size: 2500
Number of inputs: 4
```

### Step 2: Understand the Task

The goal of scientific symbolic regression is to **discover mathematical equations from data** . For the bacterial growth dataset, we need to find a function that predicts growth rate.

**Function signature:** `equation(b, s, temp, pH, params) -> growth_rate`

**Input variables:**

- `b`: Population density
- `s`: Substrate concentration
- `temp`: Temperature
- `pH`: pH level
- `params`: Array of optimizable constants (params[0] to params[9])

**Evaluation process:**

1. You provide the equation structure (e.g., `params[0] * s / (params[1] + s)`)
2. The framework automatically optimizes parameter values using `scipy.optimize.minimize`
3. MSE (Mean Squared Error) on the test set is calculated as fitness (lower is better)

### Step 3: Test with Initial Solution

```python
# Get initial solution (simple linear model)
init_sol = task.make_init_sol_wo_other_info()

print("Initial solution code:")
print(init_sol.sol_string)

# Evaluate it
result = task.evaluate_code(init_sol.sol_string)
print(f"Score: {result.score:.6f}")
print(f"Test MSE: {result.additional_info['test_mse']:.6f}")
```

**Output:**
```python
Initial solution code:
import numpy as np

def equation(b, s, temp, pH, params):
    """Linear baseline model."""
    return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]

Score: 0.017200
Test MSE: 0.017200
```

### Step 4: Try a Custom Initial Solution

You can provide a custom initial equation as the starting point for evolution. For example, here's a more complex model based on biological mechanisms:

```python
custom_code = '''import numpy as np

def equation(b, s, temp, pH, params):
    """Nonlinear bacterial growth model with biological mechanisms."""

    # Monod equation for substrate limitation
    growth_rate = params[0] * s / (params[1] + s)

    # Gaussian temperature effect
    optimal_temp = params[4]
    temp_effect = params[2] * np.exp(-params[3] * (temp - optimal_temp)**2)

    # Gaussian pH effect
    optimal_pH = params[7]
    pH_effect = params[5] * np.exp(-params[6] * (pH - optimal_pH)**2)

    # Logistic growth with carrying capacity
    carrying_capacity = params[9]
    density_limit = params[8] * (1 - b / carrying_capacity)

    return growth_rate * temp_effect * pH_effect * density_limit
'''

result = task.evaluate_code(custom_code)
print(f"Custom model score: {result.score:.6f}")
print(f"Test MSE: {result.additional_info['test_mse']:.6f}")
```

**Output:**
```
Custom model score: 0.021515
Test MSE: 0.021515
```

!!! note "About Initial Solutions"
    Note: Any custom equation you write here serves only as an **initialization solution**. The evolutionary algorithm will use the LLM to generate and improve equations starting from this point. The final evolutionary results depend on the chosen evolution method and its internal prompt design.

### Step 5: Run Evolution with EvoEngineer

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi
import os

# Create interface for EvoEngineer
interface = EvoEngineerPythonInterface(task)

# Configure LLM API
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key-here",
    model="gpt-4o"
)

# Run evolution
result = evotoolkit.solve(
    interface=interface,
    output_path='./scientific_regression_results',
    running_llm=llm_api,
    max_generations=5,
    pop_size=10
)

print(f"Best solution found!")
print(f"Score: {result['best_solution'].evaluation_res.score:.6f}")
print(f"Code:\n{result['best_solution'].sol_string}")
```

!!! tip "Try Other Algorithms"
    EvoToolkit supports multiple evolution algorithms. Simply swap the Interface:

    ```python
    # Use EoH
    from evotoolkit.task.python_task import EoHPythonInterface
    interface = EoHPythonInterface(task)

    # Use FunSearch
    from evotoolkit.task.python_task import FunSearchPythonInterface
    interface = FunSearchPythonInterface(task)
    ```

    Then use the same `evotoolkit.solve()` call to run evolution. Different algorithms may perform better on different tasks - try multiple and compare.

---

## Customizing Evolution Behavior

The quality of the evolutionary process is primarily controlled by the **evolution method** and its internal **prompt design**. If you want to improve results:

- **Adjust prompts**: Inherit existing Interface classes and customize LLM prompts
- **Develop new algorithms**: Create brand new evolutionary strategies and operators

!!! tip "Learn More"
    These are universal techniques applicable to all tasks. For detailed tutorials, see:

    - **[Customizing Evolution Methods](../customization/customizing-evolution.md)** - How to modify prompts and develop new algorithms
    - **[Advanced Usage](../advanced-overview.md)** - More advanced configuration options

**Quick Example - Customize prompt for scientific regression:**

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface

class ScientificRegressionInterface(EvoEngineerPythonInterface):
    """Interface optimized for scientific equation discovery, with custom mutation prompt"""

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """Customize the mutation operator prompt to emphasize physical/biological principles"""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            prompt = f"""You are an expert in scientific equation discovery.

Task: {task_description}

Current best equation (score: {current_best_sol.evaluation_res.score:.5f}):
{current_best_sol.sol_string}

Requirements: Generate an improved equation based on known physical/biological principles
(e.g., Monod equation, Arrhenius equation). Ensure numerical stability and model parsimony.

Output format:
- name: equation name
- code: Python code
- thought: improvement rationale
"""
            return [{"role": "user", "content": prompt}]

        # init and crossover operators use parent class default prompts
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# Use custom Interface
interface = ScientificRegressionInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

!!! note "About EvoEngineer Operators"
    EvoEngineer uses three operators: **init** (initialization), **mutation** (mutation), **crossover** (crossover).
    The parent class `EvoEngineerPythonInterface` already defines these operators and default prompts.
    You only need to override `get_operator_prompt()` to customize specific operator prompts - others will automatically use the default implementation.

For complete customization tutorials and more examples, see [Customizing Evolution Methods](../customization/customizing-evolution.md).

---

## Understanding Evaluation

### How Scoring Works

1. **Parameter Optimization**: Your equation structure is evaluated by optimizing parameters using `scipy.optimize.minimize` with BFGS method
2. **MSE Calculation**: Mean Squared Error between predictions and ground truth
3. **Fitness**: Negative MSE (higher is better, so lower MSE = higher fitness)

### Evaluation Output

```python
result = task.evaluate_code(code)

if result.valid:
    print(f"Score: {result.score}")                           # Higher is better
    print(f"Train MSE: {result.additional_info['train_mse']}")  # On training data
    print(f"Test MSE: {result.additional_info['test_mse']}")    # On test data (used for fitness)
else:
    print(f"Error: {result.additional_info['error']}")
```

---

## Next Steps

### Explore different tasks and methods

- Try different datasets (oscillator1, oscillator2, stressstrain)
- Compare results across evolution methods (EvoEngineer, EoH, FunSearch)
- Visualize predictions vs ground truth

### Customize and improve the evolution process

- Inspect prompt designs in existing Interface classes
- Inherit and override Interface to customize prompts
- Design specialized prompts for different operators (init/mutation/crossover)
- If needed, develop brand new evolution algorithms

### Learn more

- [Customizing Evolution Methods](../customization/customizing-evolution.md) - Deep dive into prompt customization and algorithm development
- [Advanced Usage](../advanced-overview.md) - Advanced configurations and techniques
- [API Reference](../../api/index.md) - Complete API documentation
- [Development Docs](../../development/contributing.md) - Contributing new methods and features
