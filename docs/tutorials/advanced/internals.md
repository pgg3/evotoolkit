# Algorithm Internals

Access and analyze the internal state of evolutionary algorithms.

---

## Overview

EvoToolkit's low-level API provides full access to algorithm internals, allowing you to:
- Inspect evolution history
- Access solution populations
- Extract metrics and statistics
- Plot evolution progress

---

## Accessing Run State

All algorithms store their internal state in `run_state_dict`:

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

algorithm = EvoEngineer(config)
algorithm.run()

# Access run state
run_state = algorithm.run_state_dict

# Get all solutions history
all_solutions = run_state.sol_history

# Get current population
current_population = run_state.population
```

---

## Inspecting Evolution History

### Get All Solutions

```python
# All solutions ever generated (including invalid ones)
all_solutions = algorithm.run_state_dict.sol_history

print(f"Total solutions generated: {len(all_solutions)}")
```

---

### Filter Valid Solutions

```python
# Only valid solutions
valid_solutions = [
    sol for sol in all_solutions
    if sol.evaluation_res.valid
]

print(f"Valid solutions: {len(valid_solutions)}")
print(f"Success rate: {len(valid_solutions) / len(all_solutions) * 100:.1f}%")
```

---

### Get Score History

```python
# Extract scores (higher is better)
score_history = [
    sol.evaluation_res.score
    for sol in all_solutions
    if sol.evaluation_res.valid
]

print(f"Best score: {max(score_history)}")
print(f"Average score: {sum(score_history) / len(score_history):.4f}")
print(f"Score improvement: {max(score_history) - score_history[0]:.4f}")
```

---

### Get Best Solution

```python
# Method 1: Using built-in helper
best_solution = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)

# Method 2: Manual search
best_solution = max(
    all_solutions,
    key=lambda s: s.evaluation_res.score if s.evaluation_res.valid else float('-inf')
)

print(f"Best score: {best_solution.evaluation_res.score}")
print(f"Best code:\n{best_solution.sol_string}")
```

---

## Solution Object Structure

Each solution contains detailed information:

```python
solution = all_solutions[0]

# Core attributes
solution.sol_string          # The actual code/solution string
solution.evaluation_res      # Evaluation result object
solution.other_info         # Additional metadata dictionary

# Evaluation result
eval_res = solution.evaluation_res
eval_res.valid              # Boolean: is solution valid?
eval_res.score              # Float: fitness score (higher = better)
eval_res.error_message      # String: error if invalid
eval_res.metadata           # Dict: additional evaluation info

# Example: print solution details
for i, sol in enumerate(all_solutions[:5]):
    print(f"\nSolution {i+1}:")
    print(f"  Valid: {sol.evaluation_res.valid}")
    print(f"  Score: {sol.evaluation_res.score:.4f}")
    if not sol.evaluation_res.valid:
        print(f"  Error: {sol.evaluation_res.error_message}")
```

---

## Plotting Evolution Progress

### Score Over Time

```python
import matplotlib.pyplot as plt

# Get valid scores in order
scores = [
    sol.evaluation_res.score
    for sol in all_solutions
    if sol.evaluation_res.valid
]

plt.figure(figsize=(10, 6))
plt.plot(scores, marker='o', alpha=0.6, linewidth=1, markersize=4)
plt.xlabel('Solution Index', fontsize=12)
plt.ylabel('Score (Higher = Better)', fontsize=12)
plt.title('Evolution Progress', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results/evolution_progress.png', dpi=300)
plt.show()
```

---

### Best Score by Generation

```python
import matplotlib.pyplot as plt
import numpy as np

# Group solutions by generation
generations = {}
for sol in all_solutions:
    if sol.evaluation_res.valid:
        gen = sol.other_info.get('generation', 0)
        if gen not in generations:
            generations[gen] = []
        generations[gen].append(sol.evaluation_res.score)

# Get best score per generation
gen_numbers = sorted(generations.keys())
best_scores = [max(generations[gen]) for gen in gen_numbers]
avg_scores = [np.mean(generations[gen]) for gen in gen_numbers]

plt.figure(figsize=(10, 6))
plt.plot(gen_numbers, best_scores, 'g-o', label='Best Score', linewidth=2)
plt.plot(gen_numbers, avg_scores, 'b--s', label='Average Score', linewidth=2)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Score by Generation', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results/score_by_generation.png', dpi=300)
plt.show()
```

---

### Success Rate Analysis

```python
import matplotlib.pyplot as plt

# Calculate success rate by generation
success_rates = []
for gen in gen_numbers:
    total = len([s for s in all_solutions if s.other_info.get('generation') == gen])
    valid = len(generations.get(gen, []))
    success_rates.append(valid / total * 100 if total > 0 else 0)

plt.figure(figsize=(10, 6))
plt.bar(gen_numbers, success_rates, alpha=0.7, color='steelblue')
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.title('Solution Validity by Generation', fontsize=14)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./results/success_rate.png', dpi=300)
plt.show()
```

---

## Analyzing Solution Diversity

### Code Length Distribution

```python
import matplotlib.pyplot as plt

# Get code lengths
code_lengths = [
    len(sol.sol_string)
    for sol in all_solutions
    if sol.evaluation_res.valid
]

plt.figure(figsize=(10, 6))
plt.hist(code_lengths, bins=20, alpha=0.7, color='coral', edgecolor='black')
plt.xlabel('Code Length (characters)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Solution Code Length Distribution', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./results/code_length_dist.png', dpi=300)
plt.show()
```

---

### Score Distribution

```python
import matplotlib.pyplot as plt

scores = [
    sol.evaluation_res.score
    for sol in all_solutions
    if sol.evaluation_res.valid
]

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Score Distribution', fontsize=14)
plt.axvline(max(scores), color='r', linestyle='--', linewidth=2, label=f'Best: {max(scores):.4f}')
plt.axvline(np.mean(scores), color='b', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./results/score_distribution.png', dpi=300)
plt.show()
```

---

## Extracting Metrics

### Comprehensive Statistics

```python
import numpy as np

def compute_statistics(all_solutions):
    """Compute comprehensive evolution statistics"""

    valid_solutions = [s for s in all_solutions if s.evaluation_res.valid]
    scores = [s.evaluation_res.score for s in valid_solutions]

    stats = {
        'total_solutions': len(all_solutions),
        'valid_solutions': len(valid_solutions),
        'success_rate': len(valid_solutions) / len(all_solutions) * 100,
        'best_score': max(scores) if scores else None,
        'worst_score': min(scores) if scores else None,
        'mean_score': np.mean(scores) if scores else None,
        'median_score': np.median(scores) if scores else None,
        'std_score': np.std(scores) if scores else None,
        'score_range': max(scores) - min(scores) if scores else None,
    }

    return stats

stats = compute_statistics(all_solutions)

print("Evolution Statistics:")
print(f"  Total solutions: {stats['total_solutions']}")
print(f"  Valid solutions: {stats['valid_solutions']}")
print(f"  Success rate: {stats['success_rate']:.1f}%")
print(f"\nScore Statistics:")
print(f"  Best: {stats['best_score']:.4f}")
print(f"  Worst: {stats['worst_score']:.4f}")
print(f"  Mean: {stats['mean_score']:.4f}")
print(f"  Median: {stats['median_score']:.4f}")
print(f"  Std Dev: {stats['std_score']:.4f}")
print(f"  Range: {stats['score_range']:.4f}")
```

---

### Export to DataFrame

```python
import pandas as pd

# Convert solutions to DataFrame for analysis
data = []
for i, sol in enumerate(all_solutions):
    data.append({
        'index': i,
        'valid': sol.evaluation_res.valid,
        'score': sol.evaluation_res.score if sol.evaluation_res.valid else None,
        'generation': sol.other_info.get('generation', -1),
        'code_length': len(sol.sol_string),
        'error': sol.evaluation_res.error_message if not sol.evaluation_res.valid else None
    })

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('./results/evolution_data.csv', index=False)

# Quick analysis
print(df.describe())
print("\nScore by generation:")
print(df.groupby('generation')['score'].agg(['mean', 'max', 'count']))
```

---

## Next Steps

- Learn [Debugging & Profiling](debugging.md) to troubleshoot issues
- Review [Low-Level API](low-level-api.md) for more control options
- Check [Configuration](configuration.md) for parameter tuning

---

## Resources

- [Matplotlib Documentation](https://matplotlib.org/) - Plotting library
- [Pandas Documentation](https://pandas.pydata.org/) - Data analysis
- [NumPy Documentation](https://numpy.org/) - Numerical computing