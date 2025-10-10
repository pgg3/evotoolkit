# Tasks API

Tasks define optimization problems and how to evaluate candidate solutions.

---

## Overview

EvoToolkit provides three categories of tasks:

- **Python Tasks** - Optimize Python code functions
- **String Tasks** - Optimize text/string solutions (e.g., prompts)
- **CUDA Tasks** - Optimize GPU kernel code

---

## Python Tasks

### PythonTask

Base class for Python code optimization tasks.

See the dedicated page: [PythonTask](tasks/python/python-task.md).

---

### ScientificRegressionTask

Scientific symbolic regression for discovering mathematical equations from data.

See the dedicated page: [ScientificRegressionTask](tasks/python/scientific-regression.md).

---

### AdversarialAttackTask

Evolve adversarial attack algorithms for black-box models.

**Usage:**

```python
from evotoolkit.task.python_task.adversarial_attack import AdversarialAttackTask

# Create task with mock evaluation
task = AdversarialAttackTask(
    model=None,  # Optional: PyTorch model
    test_loader=None,  # Optional: test data loader
    attack_steps=1000,
    n_test_samples=10,
    timeout_seconds=300.0,
    use_mock=True  # Use mock evaluation for testing
)

# Evaluate attack code
code = '''
def draw_proposals(x, num_proposals, step_size):
    # Generate adversarial proposal samples
    proposals = ...
    return proposals
'''

result = task.evaluate_code(code)
print(f"Score: {result.score}")  # Negative L2 distance (higher is better)
```

**Parameters:**

- `model` (`any`, optional): Target model to attack. If None, uses mock evaluation.
- `test_loader` (`any`, optional): DataLoader with test samples. If None, uses mock evaluation.
- `attack_steps` (`int`): Number of attack iterations per sample (default: 1000)
- `n_test_samples` (`int`): Number of test samples to evaluate (default: 10)
- `timeout_seconds` (`float`): Execution timeout (default: 300.0)
- `use_mock` (`bool`): Use mock evaluation instead of real attack (default: False)

**Methods:**

- `evaluate_code(code: str) -> EvaluationResult`: Evaluate attack algorithm code

See [Adversarial Attack Tutorial](../tutorials/built-in/adversarial-attack.md) for details.

---

## String Tasks

### StringTask

Base class for string-based optimization tasks (e.g., prompt optimization).

**Usage:**

```python
from evotoolkit.task.string_optimization.string_task import StringTask
from evotoolkit.core import EvaluationResult, Solution

class MyStringTask(StringTask):
    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        # Evaluate string solution
        score = self.compute_score(candidate_string)
        return EvaluationResult(
            valid=True,
            score=score,
            additional_info={}
        )

    def get_base_task_description(self) -> str:
        return "Optimize a string solution..."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("initial string")
```

**Constructor:**

```python
def __init__(self, data, timeout_seconds: float = 30.0)
```

**Abstract Methods:**

- `_evaluate_string_impl(candidate_string: str) -> EvaluationResult`
- `get_base_task_description() -> str`
- `make_init_sol_wo_other_info() -> Solution`

---

### PromptOptimizationTask

Optimize LLM prompt templates to improve task performance.

**Usage:**

```python
from evotoolkit.task.string_optimization.prompt_optimization import PromptOptimizationTask

# Define test cases
test_cases = [
    {"question": "What is 2+2?", "expected": "4"},
    {"question": "What is 5*3?", "expected": "15"}
]

# Create task
task = PromptOptimizationTask(
    test_cases=test_cases,
    llm_api=my_llm_api,  # Optional if use_mock=True
    timeout_seconds=30.0,
    use_mock=True  # Use mock LLM for testing
)

# Evaluate prompt template
prompt_template = "Solve this math problem: {question}\nGive only the number."
result = task.evaluate_code(prompt_template)
print(f"Accuracy: {result.score}")  # Correctness rate (0.0 to 1.0)
```

**Parameters:**

- `test_cases` (`List[Dict[str, str]]`): Test cases with 'question' and 'expected' keys
- `llm_api` (optional): LLM API instance for testing prompts (required if `use_mock=False`)
- `timeout_seconds` (`float`): Evaluation timeout (default: 30.0)
- `use_mock` (`bool`): Use mock LLM responses for testing (default: False)

**Template Format:**

Prompt templates must contain `{question}` placeholder:

```python
# Valid templates
"Answer the question: {question}"
"Q: {question}\nA:"

# Invalid - missing placeholder
"Answer the question"  # ERROR!
```

**Methods:**

- `evaluate_code(prompt_template: str) -> EvaluationResult`: Evaluate prompt template

See [Prompt Engineering Tutorial](../tutorials/built-in/prompt-engineering.md) for details.

---

## CUDA Tasks

### CudaTask

Base class for CUDA kernel optimization tasks.

**Usage:**

```python
from evotoolkit.task.cuda_engineering import CudaTask, CudaTaskInfoMaker, Evaluator

# Create evaluator
evaluator = Evaluator(temp_path='./temp')

# Create task info
task_info = CudaTaskInfoMaker.make_task_info(
    evaluator=evaluator,
    gpu_type="RTX 4090",
    cuda_version="12.4",
    org_py_code=original_python_code,
    func_py_code=function_python_code,
    cuda_code=baseline_cuda_code
)

# Create task
task = CudaTask(data=task_info, temp_path='./temp')

# Evaluate CUDA code
eval_res = task.evaluate_code(candidate_cuda_code)
print(f"Runtime: {-eval_res.score:.4f}s")  # Score is negative runtime
```

**Constructor:**

```python
def __init__(self, data, temp_path=None, fake_mode: bool = False)
```

**Parameters:**

- `data` (`dict`): Task info from `CudaTaskInfoMaker.make_task_info()`
- `temp_path` (`str`, optional): Temporary path for CUDA compilation
- `fake_mode` (`bool`): Skip actual CUDA evaluation (default: False)

**Methods:**

- `evaluate_code(code: str) -> EvaluationResult`: Evaluate CUDA kernel code and return result with negative runtime as score (higher score = faster kernel)

**Note:** CUDA tasks require the `cuda_engineering` extra:

```bash
pip install evotoolkit[cuda_engineering]
```

See [CUDA Task Tutorial](../tutorials/built-in/cuda-task.md) for a complete example.

---

## Data Management

Datasets are automatically downloaded from GitHub releases when first accessed.

### Python API

```python
from evotoolkit.data import get_dataset_path, list_available_datasets

# Get dataset path (auto-downloads if not present)
base_dir = get_dataset_path('scientific_regression')

# Access specific dataset
bactgrow_path = base_dir / 'bactgrow'
train_csv = bactgrow_path / 'train.csv'

# Use custom directory
base_dir = get_dataset_path('scientific_regression', data_dir='./my_data')

# List available datasets in a category
datasets = list_available_datasets('scientific_regression')
print(datasets.keys())  # dict_keys(['bactgrow', 'oscillator1', 'oscillator2', 'stressstrain'])
```

**Available Functions:**

- `get_dataset_path(category, data_dir=None)` - Get dataset path, auto-download if needed
- `list_available_datasets(category)` - List all datasets in a category

**Default Location:** `~/.evotool/data/`

---

## Creating Custom Tasks

### Python Task Example

```python
from evotoolkit.task.python_task import PythonTask
from evotoolkit.core import EvaluationResult, Solution

class MyOptimizationTask(PythonTask):
    """Custom optimization task"""

    def __init__(self, data, target):
        self.data = data
        self.target = target
        super().__init__(data={'data': data, 'target': target}, timeout_seconds=30.0)

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """Evaluate solution and return result (higher score is better)"""
        # 1. Execute solution code
        namespace = {}
        try:
            exec(candidate_code, namespace)
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': str(e)}
            )

        # 2. Extract function
        if 'my_function' not in namespace:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Function "my_function" not found'}
            )

        func = namespace['my_function']

        # 3. Compute fitness (negative MSE so higher is better)
        try:
            predictions = [func(x) for x in self.data]
            mse = sum((p - t)**2 for p, t in zip(predictions, self.target)) / len(self.data)
            score = -mse
            return EvaluationResult(
                valid=True,
                score=score,
                additional_info={'mse': mse}
            )
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': str(e)}
            )

    def get_base_task_description(self) -> str:
        return "Optimize a function to fit the data..."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("def my_function(x): return x")
```

See [Custom Task Tutorial](../tutorials/customization/custom-task.md) for details.

---

## Task Selection Guide

| Task Type | Recommended Class | Use Case |
|-----------|-------------------|----------|
| Scientific equation discovery | `ScientificRegressionTask` | Discover mathematical models from data |
| Adversarial attacks | `AdversarialAttackTask` | Evolve attack algorithms |
| Prompt optimization | `PromptOptimizationTask` | Optimize LLM prompts |
| Python code | `PythonTask` | General Python optimization |
| String optimization | `StringTask` | Text/configuration optimization |
| GPU kernels | `CudaTask` | CUDA performance optimization |
| Custom problems | `BaseTask` | Any other optimization problem |

---

## Next Steps

- Explore [Methods API](methods.md) for evolutionary algorithms
- Check [Interfaces API](interfaces.md) for task-method connections
- Try [Scientific Regression Tutorial](../tutorials/built-in/scientific-regression.md)
- Learn to create [Custom Tasks](../tutorials/customization/custom-task.md)
