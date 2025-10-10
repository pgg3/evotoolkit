# Custom Task Tutorial

Learn how to create your own optimization tasks in EvoToolkit.

---

## Overview

This tutorial shows you how to:

- Extend the `PythonTask` base class
- Implement custom evaluation logic
- Use your custom task with evolutionary algorithms

!!! tip "Complete Example Code"
    This tutorial provides a complete runnable example (click to view/download):

    - [:material-download: my_custom_task.py](https://github.com/pgg3/evotoolkit/blob/master/examples/custom_task/my_custom_task.py) - Complete custom task example

    Run locally:
    ```bash
    cd examples/custom_task
    python my_custom_task.py
    ```

---

## Prerequisites

- Completed the [Scientific Regression Tutorial](../built-in/scientific-regression.md)
- Understanding of Python classes and inheritance

---

## Creating a Custom Task

### Step 1: Define the Task Class

```python
from evotoolkit.task.python_task import PythonTask
from evotoolkit.core import Solution, EvaluationResult
import numpy as np

class MyOptimizationTask(PythonTask):
    """Custom task for optimizing a specific problem"""

    def __init__(self, data, target, timeout_seconds=30.0):
        """
        Initialize task with problem-specific data

        Args:
            data: Input data (NumPy array)
            target: Target output values (NumPy array)
            timeout_seconds: Code execution timeout (seconds)
        """
        self.target = target
        super().__init__(data, timeout_seconds)

    def _process_data(self, data):
        """Process input data and create task_info"""
        self.data = data
        self.task_info = {
            'data_size': len(data),
            'description': 'Function approximation task'
        }

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """Evaluate candidate code and return evaluation result"""
        # 1. Execute code
        namespace = {'np': np}
        exec(candidate_code, namespace)

        # 2. Check if function exists
        if 'my_function' not in namespace:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Function "my_function" not found'}
            )

        evolved_func = namespace['my_function']

        # 3. Compute fitness (higher score is better)
        predictions = np.array([evolved_func(x) for x in self.data])
        mse = np.mean((predictions - self.target) ** 2)
        score = -mse  # Negative MSE, higher is better

        return EvaluationResult(
            valid=True,
            score=score,
            additional_info={'mse': mse}
        )

    def get_base_task_description(self) -> str:
        """Get task description for prompt generation"""
        return """You are a function approximation expert.

Task: Create a function my_function(x) that produces outputs as close as possible to target values.

Requirements:
- Define function my_function(x: float) -> float
- Use mathematical operations: +, -, *, /, **, np.exp, np.log, np.sin, np.cos, etc.
- Ensure numerical stability

Example code:
    import numpy as np

    def my_function(x):
        return np.sin(x)
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial solution"""
        initial_code = '''import numpy as np

def my_function(x):
    """Simple linear function as baseline"""
    return x
'''
        eval_res = self.evaluate_code(initial_code)
        return Solution(
            sol_string=initial_code,
            evaluation_res=eval_res
        )
```

**Key Points:**

- Inherit from `PythonTask` instead of directly from `BaseTask`
- Implement `_evaluate_code_impl()` returning `EvaluationResult` object
- Implement `get_base_task_description()` to provide task description
- Implement `make_init_sol_wo_other_info()` to create initial solution
- Use `_process_data()` to set up `task_info`
- `score` should be higher for better solutions (use negative MSE)

---

## Step 2: Use Your Custom Task

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi
import numpy as np
import os

# Create task instance
data = np.linspace(0, 10, 50)
target = np.sin(data)  # Target: approximate sine function

task = MyOptimizationTask(data, target)

# Create interface
interface = EvoEngineerPythonInterface(task)

# Setup LLM
llm_api = HttpsApi(
    api_url=os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
    key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
    model="gpt-4o"
)

# Solve
result = evotoolkit.solve(
    interface=interface,
    output_path='./results/custom_task',
    running_llm=llm_api,
    max_generations=10
)

print(f"Best score: {result.evaluation_res.score:.4f}")
print(f"Best MSE: {result.evaluation_res.additional_info['mse']:.4f}")
```

---

## Example: String Matching Task

```python
from evotoolkit.task.python_task import PythonTask
from evotoolkit.core import Solution, EvaluationResult

class StringMatchTask(PythonTask):
    """Task to evolve a function that generates a target string"""

    def __init__(self, target_string, timeout_seconds=30.0):
        self.target = target_string
        super().__init__(data={'target': target_string}, timeout_seconds=timeout_seconds)

    def _process_data(self, data):
        """Process input data"""
        self.data = data
        self.task_info = {
            'target': self.target,
            'target_length': len(self.target)
        }

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """Evaluate code"""
        namespace = {}
        exec(candidate_code, namespace)

        if 'generate_string' not in namespace:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Function "generate_string" not found'}
            )

        try:
            generated = namespace['generate_string']()
            # Edit distance lower is better, so use negative value as score
            distance = self.levenshtein_distance(generated, self.target)
            score = -distance  # Higher is better

            return EvaluationResult(
                valid=True,
                score=score,
                additional_info={'distance': distance, 'generated': generated}
            )
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': str(e)}
            )

    def levenshtein_distance(self, s1, s2):
        """Compute Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_base_task_description(self) -> str:
        """Task description"""
        return f"""You are a string generation expert.

Task: Create a function generate_string() that generates the target string "{self.target}".

Requirements:
- Define function generate_string() -> str
- Function should return a string as close as possible to the target string

Example code:
    def generate_string():
        return "Hello, World!"
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial solution"""
        initial_code = f'''def generate_string():
    """Initial simple implementation"""
    return ""
'''
        eval_res = self.evaluate_code(initial_code)
        return Solution(
            sol_string=initial_code,
            evaluation_res=eval_res
        )
```

**Usage:**

```python
task = StringMatchTask("Hello, EvoToolkit!")
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface, './results', llm_api)
print(f"Generated string: {result.evaluation_res.additional_info['generated']}")
```

---

## Best Practices

### 1. Robust Error Handling

```python
def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
    """Implement robust error handling in _evaluate_code_impl"""
    try:
        # Execution and evaluation logic
        namespace = {}
        exec(candidate_code, namespace)
        # ... evaluation logic ...

        return EvaluationResult(
            valid=True,
            score=score,
            additional_info={}
        )
    except SyntaxError as e:
        return EvaluationResult(
            valid=False,
            score=float('-inf'),
            additional_info={'error': f'Syntax error: {str(e)}'}
        )
    except Exception as e:
        return EvaluationResult(
            valid=False,
            score=float('-inf'),
            additional_info={'error': f'Evaluation error: {str(e)}'}
        )
```

**Note:** PythonTask's parent method `evaluate_code()` already provides timeout control. Set the `timeout_seconds` parameter in the constructor.

### 2. Validate Solution Output

```python
def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
    """Validate function output type and range"""
    namespace = {}
    exec(candidate_code, namespace)

    evolved_func = namespace['my_function']
    result = evolved_func(test_input)

    # Validate type
    if not isinstance(result, (int, float, np.ndarray)):
        return EvaluationResult(
            valid=False,
            score=float('-inf'),
            additional_info={'error': 'Invalid output type'}
        )

    # Validate range
    if isinstance(result, np.ndarray):
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Output contains NaN or Inf'}
            )

    # Compute fitness
    score = -abs(result - expected)  # Negative error, higher is better
    return EvaluationResult(valid=True, score=score, additional_info={})
```

### 3. Store Task Metadata in task_info

```python
def _process_data(self, data):
    """Store important task metadata in task_info"""
    self.data = data
    self.task_info = {
        'data_size': len(data),
        'input_dim': data.shape[1] if len(data.shape) > 1 else 1,
        'description': 'Custom optimization task',
        'metric': 'MSE',
        # Other useful metadata...
    }
```

---

## Advanced: Custom Interface

If you need more fine-grained control, you can customize interfaces for different evolutionary methods. Different methods (such as EvoEngineer, FunSearch, EoH) have their own interface implementations, which control prompt generation, LLM response parsing, and other behaviors.

For details on how to customize evolutionary methods and interfaces, please refer to the [Customizing Evolution Methods Tutorial](customizing-evolution.md).

---

## Complete Example

See `examples/custom_task/my_custom_task.py` for a complete runnable example.

---

## Next Steps

- Try the [CUDA Task Tutorial](../built-in/cuda-task.md) for GPU optimization
- Explore [Advanced Usage](../advanced-overview.md) for low-level API
- Check the [API Reference](../../api/tasks.md) for Task class details
