# Exploring the Results

After running, check the `./results/` directory:

---

## Results Directory Structure

```
results/
├── run_state.json              # Run state and statistics
├── history/                    # Historical records
│   ├── gen_-1.json            # Initial population
│   ├── gen_1.json             # All solutions from generation 1
│   ├── gen_2.json             # All solutions from generation 2
│   └── ...
└── summary/                    # Summary information
    ├── usage_history.json     # LLM usage statistics
    └── best_per_generation.json  # Best solutions per generation (if any)
```

---

## Analyzing Results Programmatically

Each `gen_N.json` file contains all solutions, evaluation results, and statistics for that generation. You can load and analyze these results programmatically:

```python
import json

# Load history for a specific generation
with open('./results/history/gen_1.json', 'r') as f:
    gen_1 = json.load(f)

# View all solutions from that generation
for sol in gen_1['solutions']:
    print(f"Score: {sol['evaluation_res']['score']}")
    print(f"Solution:\n{sol['sol_string']}\n")
```

---

Next: [Try Different Algorithms](try-algorithms.md)
