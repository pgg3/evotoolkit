# Debugging & Profiling

Debug issues and optimize performance of your evolutionary workflows.

---

## Enabling Verbose Logging

### Basic Verbose Mode

```python
from evotoolkit.evo_method.evoengineer import EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    verbose=True  # Enable verbose logging
)

algorithm = EvoEngineer(config)
algorithm.run()
```

**Output Example:**
```
Generation 1/10:
  - Generated 12 solutions
  - Valid solutions: 8
  - Best score: 0.245
  - Average score: 0.512
  - Elites preserved: 2

Generation 2/10:
  - Generated 12 solutions
  - Valid solutions: 10
  - Best score: 0.189
  - Average score: 0.431
  - Elites preserved: 2
...
```

---

## Saving Intermediate Solutions

### Save All Generations

```python
config = EvoEngineerConfig(
    # ... other params
    save_all_generations=True  # Save solutions from each generation
)
```

**Directory Structure:**
```
results/
├── generation_1/
│   ├── solution_1.py
│   ├── solution_2.py
│   └── ...
├── generation_2/
│   ├── solution_1.py
│   └── ...
├── ...
└── best_solution.py
```

This allows you to:
- Inspect failed solutions
- Debug evaluation errors
- Analyze solution evolution
- Recover from crashes

---

## Inspecting LLM Interactions

### Enable LLM Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./results/llm_debug.log'),
        logging.StreamHandler()
    ]
)

# Enable LLM logger
logger = logging.getLogger('evotoolkit.llm')
logger.setLevel(logging.DEBUG)

# Now run algorithm - all LLM interactions will be logged
algorithm.run()
```

**Log Output Example:**
```
2024-01-15 10:23:45 - evotoolkit.llm - DEBUG - Sending prompt to LLM:
  --- PROMPT START ---
  You are an expert Python programmer...
  --- PROMPT END ---

2024-01-15 10:23:52 - evotoolkit.llm - DEBUG - Received LLM response:
  --- RESPONSE START ---
  def target_function(x):
      return x ** 2 + 2 * x + 1
  --- RESPONSE END ---

2024-01-15 10:23:52 - evotoolkit.llm - DEBUG - Token usage: 245 input, 67 output
```

---

### Save Prompts and Responses

```python
class DebugInterface(EvoEngineerPythonInterface):
    def __init__(self, task):
        super().__init__(task)
        self.prompt_history = []

    def query_llm(self, prompt):
        response = super().query_llm(prompt)

        # Save for debugging
        self.prompt_history.append({
            'prompt': prompt,
            'response': response,
            'timestamp': time.time()
        })

        # Save to file
        with open('./results/llm_history.json', 'w') as f:
            json.dump(self.prompt_history, f, indent=2)

        return response
```

---

## Performance Profiling

### Time Profiling

```python
import time

start_time = time.time()

algorithm = EvoEngineer(config)
algorithm.run()

elapsed = time.time() - start_time
print(f"\nTotal optimization time: {elapsed:.2f} seconds")
print(f"Time per generation: {elapsed / config.max_generations:.2f} seconds")

# Detailed timing (if available)
if hasattr(algorithm.run_state_dict, 'metadata'):
    gen_times = algorithm.run_state_dict.metadata.get('generation_times', [])
    for i, t in enumerate(gen_times):
        print(f"Generation {i+1}: {t:.2f}s")
```

---

### Detailed Profiling with cProfile

```python
import cProfile
import pstats
from pstats import SortKey

# Profile the run
profiler = cProfile.Profile()
profiler.enable()

algorithm = EvoEngineer(config)
algorithm.run()

profiler.disable()

# Save and analyze results
stats = pstats.Stats(profiler)
stats.sort_stats(SortKey.CUMULATIVE)

# Print top 20 time consumers
print("\nTop 20 time-consuming functions:")
stats.print_stats(20)

# Save to file
with open('./results/profile_stats.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats()
```

---

### Memory Profiling

```python
import tracemalloc

# Start memory tracking
tracemalloc.start()

algorithm = EvoEngineer(config)
algorithm.run()

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"\nCurrent memory usage: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

# Get top memory allocations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("\nTop 10 memory allocations:")
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

---

## Common Debugging Scenarios

### Issue: Solutions Not Improving

**Diagnosis:**
```python
# Check score progression
scores = [s.evaluation_res.score for s in all_solutions if s.evaluation_res.valid]
print(f"First 5 scores: {scores[:5]}")
print(f"Last 5 scores: {scores[-5:]}")

# Check if stuck in local optimum
if len(set(scores[-10:])) == 1:
    print("Warning: Scores not changing - may be stuck!")
```

**Solutions:**
- Increase population diversity (larger `pop_size`)
- Increase sampling (larger `max_sample_nums`)
- Adjust LLM temperature (higher for more exploration)
- Check if task evaluation is working correctly

---

### Issue: Many Invalid Solutions

**Diagnosis:**
```python
valid_count = sum(1 for s in all_solutions if s.evaluation_res.valid)
total_count = len(all_solutions)
success_rate = valid_count / total_count * 100

print(f"Success rate: {success_rate:.1f}%")

# Check error messages
errors = [s.evaluation_res.error_message for s in all_solutions if not s.evaluation_res.valid]
from collections import Counter
print("\nMost common errors:")
for error, count in Counter(errors).most_common(5):
    print(f"  {count}x: {error[:100]}...")
```

**Solutions:**
- Improve prompt clarity
- Add more examples to prompts
- Relax task constraints
- Check task evaluation logic

---

### Issue: Slow Performance

**Diagnosis:**
```python
import time

# Time each component
start = time.time()
algorithm = EvoEngineer(config)
init_time = time.time() - start

start = time.time()
algorithm.run()
run_time = time.time() - start

print(f"Initialization: {init_time:.2f}s")
print(f"Execution: {run_time:.2f}s")
print(f"Per generation: {run_time / config.max_generations:.2f}s")
```

**Solutions:**
- Increase parallelism (`num_samplers`, `num_evaluators`)
- Reduce `max_sample_nums`
- Use faster LLM model
- Optimize task evaluation code

---

## Custom Algorithm Implementation

For advanced users who want to implement their own evolutionary algorithms:

```python
from evotoolkit.core import BaseMethod, BaseConfig, Solution

class MyCustomAlgorithm(BaseMethod):
    """Custom evolutionary algorithm implementation"""

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.population = []

    def run(self):
        """Main evolution loop"""
        # Initialize population
        self.population = self.initialize_population()

        for generation in range(self.config.max_generations):
            print(f"\nGeneration {generation + 1}/{self.config.max_generations}")

            # Generate new solutions using LLM
            new_solutions = self.generate_solutions()

            # Evaluate solutions
            for solution in new_solutions:
                eval_res = self.config.interface.task.evaluate_code(solution.sol_string)
                solution.evaluation_res = eval_res

            # Update population
            self.population = self.select(self.population + new_solutions)

            # Log progress
            valid_pop = [s for s in self.population if s.evaluation_res.valid]
            if valid_pop:
                best = max(valid_pop, key=lambda s: s.evaluation_res.score)
                print(f"  Best score: {best.evaluation_res.score:.4f}")

        # Save results
        self.save_results()

    def initialize_population(self):
        """Generate initial population"""
        initial_solutions = []

        # Use LLM to generate initial solutions
        for i in range(self.config.pop_size):
            prompt = self.config.interface.get_init_prompt()
            response = self.config.running_llm.query(prompt)

            solution = Solution(sol_string=response)
            eval_res = self.config.interface.task.evaluate_code(response)
            solution.evaluation_res = eval_res

            initial_solutions.append(solution)

        return initial_solutions

    def generate_solutions(self):
        """Generate new solutions for current generation"""
        new_solutions = []

        # Example: mutation
        for parent in self.population[:3]:  # Take top 3
            prompt = self.config.interface.get_mutation_prompt(parent)
            response = self.config.running_llm.query(prompt)

            solution = Solution(sol_string=response)
            new_solutions.append(solution)

        return new_solutions

    def select(self, solutions):
        """Select best solutions for next generation"""
        # Filter valid solutions
        valid = [s for s in solutions if s.evaluation_res.valid]

        # Sort by score (higher is better)
        valid.sort(key=lambda s: s.evaluation_res.score, reverse=True)

        # Keep top pop_size solutions
        return valid[:self.config.pop_size]

    def save_results(self):
        """Save final results"""
        best = max(self.population, key=lambda s: s.evaluation_res.score)

        with open(f'{self.config.output_path}/best_solution.py', 'w') as f:
            f.write(best.sol_string)

        print(f"\nOptimization complete!")
        print(f"Best score: {best.evaluation_res.score:.4f}")
```

**Usage:**
```python
algorithm = MyCustomAlgorithm(config)
algorithm.run()
```

---

## Debugging Checklist

When things go wrong, check:

- [ ] Task evaluation function works correctly
- [ ] LLM API is responding (check logs)
- [ ] Prompts are clear and contain examples
- [ ] Solutions are being generated (check `sol_history`)
- [ ] Scores are being computed correctly
- [ ] Configuration parameters are reasonable
- [ ] Sufficient resources (API rate limits, memory)
- [ ] Output directory is writable

---

## Next Steps

- Review [Algorithm Internals](internals.md) for analysis techniques
- Check [Configuration](configuration.md) for parameter tuning
- Explore [Low-Level API](low-level-api.md) for more control

---

## Resources

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [cProfile Documentation](https://docs.python.org/3/library/profile.html)
- [tracemalloc Documentation](https://docs.python.org/3/library/tracemalloc.html)
- [Performance Profiling Guide](https://docs.python.org/3/library/debug.html)