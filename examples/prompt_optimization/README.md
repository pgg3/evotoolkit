# Prompt Optimization Examples

This directory contains examples for optimizing LLM prompt **templates** using evolutionary algorithms.

## Overview

The `PromptOptimizationTask` is a **string optimization task** that demonstrates how to use EvoToolkit to evolve prompt templates for better LLM performance.

### Key Difference from Python Tasks

| Aspect | Python Task | String Task (Prompt Optimization) |
|--------|-------------|-----------------------------------|
| **Solution type** | Python code | String template |
| **Evolution target** | Function/algorithm | Prompt text |
| **Evaluation** | Execute code | Test template with LLM |
| **Example** | `def func(x): return x**2` | `"Solve: {question}\nAnswer:"` |

## Quick Start

### Basic Example

Run the basic example with mock LLM (no API needed):

```bash
python basic_example.py
```

**Output:**
```
Initial prompt template:
  "Answer this question: {question}"
Initial score: 100.00%

Custom template: "Solve this math problem and give only the number: {question}"
Score: 100.00%

Key difference from Python task:
- Solutions are STRING TEMPLATES (not Python code)
- Templates use {question} placeholder
- Evolution optimizes the prompt string directly
```

### With Real LLM

1. Configure your LLM API credentials directly in `basic_example.py`:
   ```python
   llm_api = HttpsApi(
       api_url="api.openai.com",  # Your API URL
       key="your-api-key-here",   # Your API key
       model="gpt-4o"
   )
   ```

2. Set `use_mock=False` to use real LLM

3. Run the example or evolution code

## Template Format

Prompt templates are strings with `{question}` placeholder:

```python
# Good examples
"Answer this question: {question}"
"Solve this math problem: {question}\nGive only the number."
"Question: {question}\nThink step by step and provide only the final answer."

# Bad examples (missing placeholder)
"Solve this problem"  # ❌ No {question} placeholder
"Answer: 42"          # ❌ No {question} placeholder
```

## Task Structure

### Creating a Task

```python
from evotoolkit.task import PromptOptimizationTask

test_cases = [
    {"question": "What is 2+2?", "expected": "4"},
    {"question": "What is 5*3?", "expected": "15"},
]

task = PromptOptimizationTask(
    test_cases=test_cases,
    use_mock=True  # or llm_api=your_api for real LLM
)
```

### Testing a Template

```python
template = "Solve: {question}\nAnswer with just the number:"
result = task.evaluate_code(template)

print(f"Score: {result.score:.2%}")  # Accuracy
print(f"Correct: {result.additional_info['correct']}/{result.additional_info['total']}")
```

### Running Evolution

```python
import evotoolkit
from evotoolkit.task import EvoEngineerStringInterface

interface = EvoEngineerStringInterface(task)

result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)

# Best template found
print(result.sol_string)
```

## Available Algorithms

```python
# EvoEngineer (recommended)
from evotoolkit.task import EvoEngineerStringInterface
interface = EvoEngineerStringInterface(task)

# EoH (Evolution of Heuristics)
from evotoolkit.task import EoHStringInterface
interface = EoHStringInterface(task)

# FunSearch
from evotoolkit.task import FunSearchStringInterface
interface = FunSearchStringInterface(task)
```

## Use Cases

This task can be adapted for various prompt optimization scenarios:

1. **Math Problems**: Optimize prompts for solving math questions
2. **Classification**: Optimize prompts for text classification
3. **Translation**: Optimize prompts for language translation
4. **Extraction**: Optimize prompts for information extraction
5. **Question Answering**: Optimize prompts for QA tasks

## Customization

### Custom Evaluation Logic

```python
class CustomPromptTask(PromptOptimizationTask):
    def _check_answer(self, response: str, expected: str) -> bool:
        # Your custom evaluation logic
        return your_check_function(response, expected)
```

### Different Test Cases

```python
test_cases = [
    {"question": "Classify: This movie is great!", "expected": "positive"},
    {"question": "Classify: This movie is terrible!", "expected": "negative"},
]
```

## Architecture

```
src/evotool/task/
├── python_task/          # Python code optimization
├── cuda_engineering/     # CUDA kernel optimization
└── string_optimization/  # String optimization (NEW!)
    ├── string_task.py
    ├── prompt_optimization/
    │   └── prompt_optimization_task.py
    └── method_interface/
        ├── evoengineer_interface.py
        ├── eoh_interface.py
        └── funsearch_interface.py
```

## Next Steps

- Check out the [tutorials](../../docs/tutorials/) for more advanced usage
- Learn about [customizing evolution](../../docs/tutorials/customizing-evolution.md)
- Explore other task types (Python, CUDA, Scientific Regression)

## Notes

- **String Task vs Python Task**: Prompt optimization is a STRING task, not a Python task
- **Template Syntax**: Must include `{question}` placeholder
- **Mock Mode**: Useful for testing without LLM API costs
- **Real LLM**: Provides actual prompt optimization results
