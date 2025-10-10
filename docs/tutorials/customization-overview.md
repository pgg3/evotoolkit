# Customization

Learn how to extend EvoToolkit to solve your specific optimization problems.

## Customization Options

### Custom Tasks
**[→ Start Tutorial](customization/custom-task.md)**

Create your own optimization tasks for domain-specific problems.

**You'll Learn:**
- Extending the `Task` base class
- Implementing custom evaluation logic
- Defining solution spaces
- Integrating with evolutionary algorithms

**Prerequisites:** Basic EvoToolkit knowledge (Scientific Regression tutorial)

---

### Customizing Evolution Methods
**[→ Start Tutorial](customization/customizing-evolution.md)**

Learn how to customize evolutionary behavior by modifying prompts or developing new algorithms.

**You'll Learn:**
- Understanding the Interface architecture
- Customizing LLM prompts to improve results
- Designing task-specific prompts
- Developing brand new evolution algorithms (advanced)
- Implementing custom strategies like temperature annealing

**Prerequisites:** Basic EvoToolkit knowledge (Scientific Regression tutorial)

---

## When to Customize

### Create a Custom Task When:
- Your problem domain isn't covered by built-in tasks
- You need specific evaluation metrics
- You want to optimize domain-specific code or structures
- You need custom constraints or validation

### Customize Evolution Methods When:
- Default prompts don't work well for your task
- You want to incorporate domain knowledge into the evolution
- You need special mutation or crossover strategies
- You want to experiment with novel evolutionary approaches

---

## Quick Start Examples

### Custom Task Example
```python
from evotoolkit.core import BaseTask, Solution

class MyCustomTask(BaseTask):
    def evaluate(self, solution: Solution) -> float:
        # Your evaluation logic here
        return fitness_score
```

### Custom Interface Example
```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface

class MyCustomInterface(EvoEngineerPythonInterface):
    def get_prompt_components(self):
        # Customize prompts for your task
        return custom_prompts
```

---

## Best Practices

1. **Start Simple:** Begin with modifying existing tasks or prompts before creating entirely new ones
2. **Test Thoroughly:** Validate your custom evaluation logic with known solutions
3. **Document Well:** Clearly document your task requirements and constraints
4. **Share Your Work:** Consider contributing your custom tasks back to the community

---

## Next Steps

After mastering customization:
- Explore [Advanced Usage](advanced-overview.md) for low-level API control
- Check the [API Reference](../api/index.md) for detailed documentation
- Share your custom tasks in [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions)