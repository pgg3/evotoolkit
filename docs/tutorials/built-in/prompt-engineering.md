# Prompt Engineering Tutorial

Learn how to use LLM-driven evolution to optimize prompt templates for better downstream task performance.

!!! note "Academic Citation"
    If you use EvoToolkit in your research, please cite:

    ```bibtex
    @software{evotoolkit2025,
      title = {EvoToolkit: LLM-Driven Evolutionary Optimization},
      author = {Guo, Ping},
      year = {2025},
      url = {https://github.com/pgg3/evotoolkit}
    }
    ```

!!! tip "Complete Example Code"
    This tutorial provides complete, runnable examples (click to view/download):

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/prompt_optimization/basic_example.py) - Basic usage with mock LLM
    - [:material-file-document: README.md](https://github.com/pgg3/evotoolkit/blob/master/examples/prompt_optimization/README.md) - Examples documentation and usage guide

    Run locally:
    ```bash
    cd examples/prompt_optimization
    python basic_example.py
    ```

---

## Overview

This tutorial demonstrates:

- Creating prompt optimization tasks
- Using LLM-driven evolution to improve prompt templates
- Testing prompts on specific downstream tasks
- Evolving high-quality prompts automatically

---

## Installation

Install EvoToolkit:

```bash
pip install evotoolkit
```

**Prerequisites:**

- Python >= 3.11
- LLM API access (OpenAI, Claude, or other compatible providers)
- Basic understanding of prompt engineering

---

## Understanding Prompt Optimization Tasks

### What is a Prompt Optimization Task?

A prompt optimization task evolves **string templates** to maximize performance on downstream tasks. Unlike Python tasks that evolve code, prompt tasks evolve prompt text directly.

| Aspect | Python Task | Prompt Task |
|--------|-------------|-------------|
| **Solution type** | Python code | String template |
| **Evolution target** | Function/algorithm | Prompt text |
| **Evaluation** | Execute code | Test template with LLM |
| **Example** | `def func(x): return x**2` | `"Solve: {question}\nAnswer:"` |

### Task Components

A prompt optimization task requires:

- **Test cases**: Question-answer pairs for evaluation
- **Template syntax**: String with `{question}` placeholder
- **LLM API**: For testing prompt templates (or use mock mode)
- **Evaluation metric**: Accuracy on test cases

---

## Creating Your First Prompt Task

### Step 1: Define Test Cases

Create test cases with questions and expected answers:

```python
test_cases = [
    {"question": "What is 2+2?", "expected": "4"},
    {"question": "What is 5*3?", "expected": "15"},
    {"question": "What is 10-7?", "expected": "3"},
    {"question": "What is 12/4?", "expected": "3"},
    {"question": "What is 7+8?", "expected": "15"},
]
```

### Step 2: Create the Task

```python
from evotoolkit.task import PromptOptimizationTask
from evotoolkit.tools.llm import HttpsApi

# Configure LLM API
llm_api = HttpsApi(
    api_url="your_api_url",  # e.g., "ai.api.example.com"
    key="your_api_key",       # Your API key
    model="gpt-4o"
)

task = PromptOptimizationTask(
    test_cases=test_cases,
    llm_api=llm_api,
    use_mock=False
)
```

### Step 3: Test Initial Template

```python
# Get initial solution
init_sol = task.make_init_sol_wo_other_info()

print(f"Initial template: {init_sol.sol_string}")
print(f"Accuracy: {init_sol.evaluation_res.score:.2%}")
print(f"Correct: {init_sol.evaluation_res.additional_info['correct']}/{init_sol.evaluation_res.additional_info['total']}")
```

**Output:**
```
Initial template: "Answer this question: {question}"
Accuracy: 100.00%
Correct: 5/5
```

### Step 4: Test Custom Templates

```python
# Test your own template
custom_template = "Solve this math problem and give only the number: {question}"
result = task.evaluate_code(custom_template)

print(f"Custom template: {custom_template}")
print(f"Accuracy: {result.score:.2%}")
print(f"Correct: {result.additional_info['correct']}/{result.additional_info['total']}")
```

---

## Running Evolution to Optimize Prompts

### Step 1: Create Interface

```python
import evotoolkit
from evotoolkit.task import EvoEngineerStringInterface

# Create interface
interface = EvoEngineerStringInterface(task)
```

### Step 2: Run Evolution

```python
# Run evolution with LLM
result = evotoolkit.solve(
    interface=interface,
    output_path='./prompt_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

print(f"Best template found: {result.sol_string}")
print(f"Accuracy: {result.evaluation_res.score:.2%}")
```

!!! tip "Try Different Algorithms"
    EvoToolkit supports multiple evolutionary algorithms for prompt optimization:

    ```python
    # Using EoH
    from evotoolkit.task import EoHStringInterface
    interface = EoHStringInterface(task)

    # Using FunSearch
    from evotoolkit.task import FunSearchStringInterface
    interface = FunSearchStringInterface(task)

    # Using EvoEngineer (default)
    from evotoolkit.task import EvoEngineerStringInterface
    interface = EvoEngineerStringInterface(task)
    ```

    Then use the same `evotoolkit.solve()` call to run evolution. Different interfaces may perform better on different tasks.

---

## Understanding Template Format

### Valid Templates

Prompt templates must include the `{question}` placeholder:

```python
# ✅ Good templates
"Answer this question: {question}"
"Solve this math problem: {question}\nGive only the number."
"Question: {question}\nThink step by step and provide only the final answer."
"Let's solve: {question}\nFirst, analyze the problem..."

# ❌ Bad templates (missing placeholder)
"Solve this problem"     # No {question} placeholder
"Answer: 42"            # No {question} placeholder
```

### Template Evolution Example

During evolution, the LLM generates improved templates:

```python
# Generation 1
"Answer: {question}"
# Accuracy: 60%

# Generation 3
"Solve this math problem: {question}\nProvide only the numerical answer."
# Accuracy: 85%

# Generation 7
"Calculate: {question}\nShow only the final number, no explanation."
# Accuracy: 100%
```

---

## Use Cases and Applications

### 1. Math Problem Solving

```python
test_cases = [
    {"question": "What is 15 * 7?", "expected": "105"},
    {"question": "What is 144 / 12?", "expected": "12"},
    # ...
]

task = PromptOptimizationTask(test_cases=test_cases, llm_api=llm_api)
```

### 2. Text Classification

```python
test_cases = [
    {"question": "This movie is amazing!", "expected": "positive"},
    {"question": "This movie is terrible!", "expected": "negative"},
    {"question": "I loved this film!", "expected": "positive"},
    # ...
]

task = PromptOptimizationTask(test_cases=test_cases, llm_api=llm_api)
```

### 3. Information Extraction

```python
test_cases = [
    {"question": "Extract the date: The meeting is on 2024-03-15", "expected": "2024-03-15"},
    {"question": "Extract the date: We'll meet on March 20th, 2024", "expected": "2024-03-20"},
    # ...
]

task = PromptOptimizationTask(test_cases=test_cases, llm_api=llm_api)
```

### 4. Translation Tasks

```python
test_cases = [
    {"question": "Translate to French: Hello", "expected": "Bonjour"},
    {"question": "Translate to French: Thank you", "expected": "Merci"},
    # ...
]

task = PromptOptimizationTask(test_cases=test_cases, llm_api=llm_api)
```

---

## Customizing Evolution Behavior

The quality of evolved prompts is controlled by the **evolution method** and its internal **prompt design**. To improve results:

- **Adjust prompts**: Inherit existing Interface classes and customize LLM prompts
- **Develop new algorithms**: Create entirely new evolutionary strategies

!!! tip "Learn More"
    These are general techniques applicable to all tasks. For detailed tutorials, see:

    - **[Customizing Evolution Methods](../customization/customizing-evolution.md)** - How to modify prompts and develop new algorithms
    - **[Advanced Usage](../advanced-overview.md)** - More advanced configuration options

**Quick Example - Custom Prompts for Prompt Optimization:**

```python
from evotoolkit.task import EvoEngineerStringInterface

class CustomPromptInterface(EvoEngineerStringInterface):
    """Interface optimized for prompt template evolution."""

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """Customize mutation prompt to emphasize clarity and structure."""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            individual = selected_individuals[0]

            prompt = f"""# Prompt Template Optimization

{task_description}

## Current Best Template
**Accuracy:** {current_best_sol.evaluation_res.score:.2%}
**Template:** {current_best_sol.sol_string}

## Template to Mutate
**Accuracy:** {individual.evaluation_res.score:.2%}
**Template:** {individual.sol_string}

## Optimization Guidelines
Focus on improving the template by:
- Adding clear instructions
- Specifying output format explicitly
- Including relevant context or examples
- Using appropriate tone and style
- Ensuring the {{question}} placeholder is preserved

Generate an improved template that increases accuracy.

## Response Format:
name: [descriptive_name]
code:
[Your improved template with {{question}} placeholder]
thought: [reasoning for changes]
"""
            return [{"role": "user", "content": prompt}]

        # Use default for other operators
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# Use custom interface
interface = CustomPromptInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./custom_results',
    running_llm=llm_api,
    max_generations=10
)
```

---

## Understanding Evaluation

### Scoring Mechanism

1. **Template Testing**: Each template is tested on all test cases
2. **LLM Response**: The LLM generates answers using the template
3. **Answer Checking**: Responses are compared to expected answers
4. **Accuracy Calculation**: Score = (correct answers) / (total test cases)

### Evaluation Output

```python
result = task.evaluate_code(template)

if result.valid:
    print(f"Accuracy: {result.score:.2%}")
    print(f"Correct: {result.additional_info['correct']}/{result.additional_info['total']}")
    print(f"Details: {result.additional_info['details']}")
else:
    print(f"Error: {result.additional_info['error_msg']}")
```

### Mock Mode for Testing

Use mock mode to test without LLM API costs:

```python
# Mock mode always returns correct answers for testing
task = PromptOptimizationTask(
    test_cases=test_cases,
    use_mock=True  # No actual LLM calls
)

# Good for:
# - Testing task setup
# - Debugging template format
# - Understanding the workflow
# - Developing custom interfaces
```

---

## Custom Evaluation Logic

For specialized tasks, you can customize answer checking:

```python
from evotoolkit.task import PromptOptimizationTask

class CustomPromptTask(PromptOptimizationTask):
    """Custom task with specialized answer checking."""

    def _check_answer(self, response: str, expected: str) -> bool:
        """Custom evaluation logic."""
        # Example: Case-insensitive comparison
        return response.strip().lower() == expected.strip().lower()

        # Example: Fuzzy matching
        # from difflib import SequenceMatcher
        # similarity = SequenceMatcher(None, response, expected).ratio()
        # return similarity > 0.8

        # Example: Regex matching
        # import re
        # return bool(re.search(expected, response))

# Use custom task
test_cases = [
    {"question": "Capital of France?", "expected": "paris"},
    # "Paris", "PARIS", "paris" all accepted
]

task = CustomPromptTask(test_cases=test_cases, llm_api=llm_api)
```

---

## Complete Example

Here's a full working example:

```python
import evotoolkit
from evotoolkit.task import PromptOptimizationTask, EvoEngineerStringInterface
from evotoolkit.tools.llm import HttpsApi

# 1. Define test cases
test_cases = [
    {"question": "What is 2+2?", "expected": "4"},
    {"question": "What is 5*3?", "expected": "15"},
    {"question": "What is 10-7?", "expected": "3"},
    {"question": "What is 12/4?", "expected": "3"},
    {"question": "What is 7+8?", "expected": "15"},
]

# 2. Configure LLM API
llm_api = HttpsApi(
    api_url="your_api_url",  # e.g., "ai.api.example.com"
    key="your_api_key",       # Your API key
    model="gpt-4o"
)

# 3. Create task
task = PromptOptimizationTask(
    test_cases=test_cases,
    llm_api=llm_api,
    use_mock=False
)

# 4. Create interface
interface = EvoEngineerStringInterface(task)

# 5. Run evolution
result = evotoolkit.solve(
    interface=interface,
    output_path='./prompt_optimization_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

# 6. Show results
print(f"Best template found:")
print(f"  {result.sol_string}")
print(f"Accuracy: {result.evaluation_res.score:.2%}")
print(f"Correct: {result.evaluation_res.additional_info['correct']}/{result.evaluation_res.additional_info['total']}")
```

---

## Next Steps

### Explore Different Optimization Strategies

- Try different evolutionary algorithms (EvoEngineer variants, EoH, FunSearch)
- Compare results across different interfaces
- Experiment with different test case sets
- Test on various downstream tasks

### Customize and Improve Evolution

- Examine prompt designs in existing Interface classes
- Inherit and override Interfaces to customize prompts
- Design specialized prompts for different task types
- Develop new evolutionary algorithms if needed

### Learn More

- [Customizing Evolution Methods](../customization/customizing-evolution.md) - Deep dive into prompt customization and algorithm development
- [Advanced Usage](../advanced-overview.md) - Advanced configuration and techniques
- [API Reference](../../api/index.md) - Complete API documentation
- [Development Docs](../../development/contributing.md) - Contribute new methods and features
