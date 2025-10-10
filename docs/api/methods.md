# Methods API

Evolutionary methods are the core algorithms that drive the optimization process.

---

## Available Algorithms

EvoToolkit provides three main evolutionary algorithms:

| Algorithm | Best For | Characteristics |
|-----------|----------|-----------------|
| **EvoEngineer** | General optimization | Versatile, robust, good default choice |
| **FunSearch** | Function discovery | Specialized for function approximation |
| **EoH** | Heuristic optimization | Fast, efficient for simple problems |

---

## EvoEngineer

See the dedicated page: [EvoEngineer](methods/evoengineer.md).

---

## FunSearch

See the dedicated page: [FunSearch](methods/funsearch.md).

---

## EoH (Evolution of Heuristics)

See the dedicated page: [EoH](methods/eoh.md).

---

## Algorithm Comparison

### When to Use Each Algorithm

**Use EvoEngineer when:**
- You have a general optimization problem
- You want a robust, well-tested algorithm
- You need good default behavior

**Use FunSearch when:**
- You're specifically looking for novel functions
- Function discovery is the primary goal
- You want to explore a diverse function space

**Use EoH when:**
- You need fast iterations
- Your problem has simple heuristics
- You want efficient resource usage

### Performance Characteristics

| Algorithm | Speed | Exploration | Exploitation | Best Fitness |
|-----------|-------|-------------|--------------|--------------|
| EvoEngineer | Medium | High | High | ⭐⭐⭐⭐⭐ |
| FunSearch | Slow | Very High | Medium | ⭐⭐⭐⭐ |
| EoH | Fast | Medium | High | ⭐⭐⭐ |

---

## Advanced: Custom Algorithms

You can implement custom evolutionary algorithms by extending `BaseMethod`:

```python
from evotoolkit.core import BaseMethod, BaseConfig

class MyCustomAlgorithm(BaseMethod):
    def run(self):
        for generation in range(self.config.max_generations):
            # 1. Generate solutions
            solutions = self.generate_solutions()

            # 2. Evaluate solutions
            for solution in solutions:
                eval_res = self.config.interface.task.evaluate_code(solution.sol_string)
                solution.evaluation_res = eval_res

            # 3. Select best solutions
            self.select_and_update_population(solutions)

    def generate_solutions(self):
        # Your custom generation logic
        pass

    def select_and_update_population(self, solutions):
        # Your custom selection logic
        pass
```

See [Advanced Usage Tutorial](../tutorials/advanced-overview.md) for details.

---

## Next Steps

- Try different algorithms with the [Tutorials](../tutorials/index.md)
- Learn about [Interfaces](interfaces.md) for connecting tasks to algorithms
- Explore [Core API](core.md) for the high-level `solve()` function
