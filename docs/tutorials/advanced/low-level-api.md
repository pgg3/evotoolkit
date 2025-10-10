# Low-Level API

Learn when and how to use the low-level API for maximum control over evolutionary optimization.

---

## High-Level vs Low-Level API

### High-Level API (Recommended for Most Users)

```python
import evotoolkit

# Simple and concise
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10
)
```

**Pros:**
- Simple and concise
- Automatic configuration
- Best practices built-in
- Less code to maintain

**Cons:**
- Less control over internals
- Limited customization
- Fixed workflow structure

**Best For:**
- Most optimization tasks
- Rapid prototyping
- Standard workflows
- Getting started quickly

---

### Low-Level API (Advanced Users)

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

# Full control over configuration
config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8,
    max_sample_nums=12,
    num_samplers=4,  # Number of parallel samplers
    num_evaluators=4,  # Number of parallel evaluators
    verbose=True
)

# Create and run algorithm
algorithm = EvoEngineer(config)
algorithm.run()

# Access internal state
best_solution = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)
all_solutions = algorithm.run_state_dict.sol_history
```

**Pros:**
- Full control over parameters
- Access to internal state
- Custom workflow integration
- Advanced debugging capabilities

**Cons:**
- More complex code
- Requires algorithm knowledge
- More maintenance burden
- Easy to misconfigure

**Best For:**
- Research and experimentation
- Custom workflow integration
- Performance optimization
- Algorithm development

---

## Using Different Algorithms

### EvoEngineer

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8,
    max_sample_nums=12,
    num_samplers=4,
    num_evaluators=4,
    verbose=True
)

algorithm = EvoEngineer(config)
algorithm.run()
```

---

### FunSearch

```python
from evotoolkit.evo_method.funsearch import FunSearch, FunSearchConfig

config = FunSearchConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_sample_nums=30,
    programs_per_prompt=2,
    num_islands=4,
    max_population_size=1000,
    num_samplers=5,
    num_evaluators=5,
    verbose=True
)

algorithm = FunSearch(config)
algorithm.run()
```

**Note:** FunSearch does not use `max_generations`. It evolves continuously based on the island model.

---

### EoH (Evolution of Heuristics)

```python
from evotoolkit.evo_method.eoh import EoH, EoHConfig

config = EoHConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20,
    selection_num=2,
    use_e2_operator=True,  # Crossover
    use_m1_operator=True,  # Mutation 1
    use_m2_operator=True,  # Mutation 2
    num_samplers=5,
    num_evaluators=5,
    verbose=True
)

algorithm = EoH(config)
algorithm.run()
```

---

## Accessing Results

### Get Best Solution

```python
algorithm.run()

# Method 1: Using built-in helper
best_solution = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)

# Method 2: Manual search
all_solutions = algorithm.run_state_dict.sol_history
valid_solutions = [s for s in all_solutions if s.evaluation_res.valid]
best_solution = max(valid_solutions, key=lambda s: s.evaluation_res.score)

print(f"Best score: {best_solution.evaluation_res.score}")
print(f"Best code:\n{best_solution.sol_string}")
```

---

### Access Evolution History

```python
# Get run state
run_state = algorithm.run_state_dict

# All solutions ever generated
all_solutions = run_state.sol_history

# Current population
current_population = run_state.population

# Score progression
scores = [
    sol.evaluation_res.score
    for sol in all_solutions
    if sol.evaluation_res.valid
]

print(f"Total solutions: {len(all_solutions)}")
print(f"Valid solutions: {len(scores)}")
print(f"Best score: {max(scores)}")
print(f"Average score: {sum(scores) / len(scores)}")
```

---

## Custom Workflow Integration

### Checkpoint and Resume

```python
import pickle

# Run for a few generations
algorithm = EvoEngineer(config)
for gen in range(5):
    algorithm.run_one_generation()

    # Save checkpoint
    with open(f'checkpoint_gen{gen}.pkl', 'wb') as f:
        pickle.dump(algorithm.run_state_dict, f)

# Later: resume from checkpoint
with open('checkpoint_gen4.pkl', 'rb') as f:
    saved_state = pickle.load(f)

algorithm.run_state_dict = saved_state
algorithm.run()  # Continue from where we left off
```

---

### Custom Stopping Criteria

```python
class CustomEvoEngineer(EvoEngineer):
    def should_stop(self):
        # Stop if we found a solution with score > 0.95
        best = self._get_best_sol(self.run_state_dict.sol_history)
        if best and best.evaluation_res.score > 0.95:
            print("Found excellent solution! Stopping early.")
            return True

        # Otherwise use default stopping criteria
        return super().should_stop()

algorithm = CustomEvoEngineer(config)
algorithm.run()
```

---

### Hybrid Algorithms

```python
# Start with EvoEngineer for exploration
config1 = EvoEngineerConfig(
    interface=interface,
    output_path='./results/phase1',
    running_llm=llm_api,
    max_generations=5,
    pop_size=10
)

algo1 = EvoEngineer(config1)
algo1.run()

# Get best solutions from phase 1
best_from_phase1 = sorted(
    algo1.run_state_dict.sol_history,
    key=lambda s: s.evaluation_res.score if s.evaluation_res.valid else float('-inf'),
    reverse=True
)[:3]

# Refine with FunSearch
config2 = FunSearchConfig(
    interface=interface,
    output_path='./results/phase2',
    running_llm=llm_api,
    max_sample_nums=50
)

algo2 = FunSearch(config2)
# Initialize with solutions from phase 1
algo2.run_state_dict.population = best_from_phase1
algo2.run()
```

---

## Next Steps

- Learn about [Algorithm Configuration](configuration.md) for detailed parameter tuning
- Explore [Algorithm Internals](internals.md) to analyze evolution behavior
- Check [Debugging & Profiling](debugging.md) for troubleshooting tips

---

## Resources

- [API Reference](../../api/methods.md) - Complete API documentation
- [Architecture Guide](../../development/architecture.md) - Understanding internals