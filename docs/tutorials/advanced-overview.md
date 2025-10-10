# Advanced Usage

Master the low-level API for maximum control and customization.

---

## Overview

These advanced tutorials cover:

- **Low-Level API** - Direct algorithm control and configuration
- **Algorithm Configuration** - Fine-tune evolution parameters
- **Algorithm Internals** - Access and analyze internal state
- **Debugging & Profiling** - Troubleshoot and optimize performance

---

## Prerequisites

- Completed [Scientific Regression](built-in/scientific-regression.md) tutorial
- Completed [Custom Task](customization/custom-task.md) tutorial
- Understanding of evolutionary algorithms

---

## Tutorials

### Low-Level API
**[→ Start Tutorial](advanced/low-level-api.md)**

Learn the difference between high-level and low-level APIs, and when to use each.

**You'll Learn:**
- High-level vs low-level API comparison
- Direct algorithm instantiation
- Accessing internal state
- Custom workflow control

**Time:** 15 minutes

---

### Algorithm Configuration
**[→ Start Tutorial](advanced/configuration.md)**

Master detailed configuration options for each evolutionary algorithm.

**You'll Learn:**
- EvoEngineer configuration parameters
- FunSearch island model setup
- EoH operator control
- Parallel execution tuning

**Time:** 20 minutes

---

### Algorithm Internals
**[→ Start Tutorial](advanced/internals.md)**

Access and analyze the internal state of evolutionary algorithms.

**You'll Learn:**
- Inspect evolution history
- Access solution populations
- Plot evolution progress
- Extract metrics and statistics

**Time:** 15 minutes

---

### Debugging & Profiling
**[→ Start Tutorial](advanced/debugging.md)**

Debug issues and optimize performance of your evolutionary workflows.

**You'll Learn:**
- Enable verbose logging
- Save intermediate solutions
- Inspect LLM prompts/responses
- Time and memory profiling
- Implement custom algorithms

**Time:** 25 minutes

---

## When to Use Advanced Features

### Use Low-Level API When:
- You need fine-grained control over the evolution process
- Default configurations don't meet your requirements
- You want to implement custom stopping criteria
- You need access to intermediate results

### Use Custom Configuration When:
- Default parameters don't work well for your task
- You want to optimize for speed or quality
- You need to tune parallel execution
- You're experimenting with algorithm variants

### Use Debugging Tools When:
- Evolution doesn't converge as expected
- You want to understand algorithm behavior
- You need to optimize resource usage
- You're developing custom algorithms

---

## Quick Reference

### Basic Low-Level Pattern
```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8,
    verbose=True
)

algorithm = EvoEngineer(config)
algorithm.run()

# Access results
best = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)
```

---

## Next Steps

After mastering advanced usage:

- Explore the [API Reference](../api/index.md) for complete documentation
- Read [Architecture Documentation](../development/architecture.md) to understand internals
- Contribute your improvements via the [Contributing Guide](../development/contributing.md)