# Algorithm Configuration

Master detailed configuration options for each evolutionary algorithm.

---

## Overview

Each evolutionary algorithm in EvoToolkit has its own configuration class with specific parameters. This tutorial covers all configuration options and how to tune them for your use case.

---

## EvoEngineer Configuration

EvoEngineer is the primary LLM-driven evolutionary algorithm.

```python
from evotoolkit.evo_method.evoengineer import EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,

    # Evolution parameters
    max_generations=20,      # Maximum number of generations
    pop_size=10,             # Population size
    max_sample_nums=15,      # Maximum samples per generation

    # Parallel control
    num_samplers=4,          # Number of parallel samplers
    num_evaluators=4,        # Number of parallel evaluators

    # Logging
    verbose=True             # Show verbose logging
)
```

### Key Parameters

**Evolution Parameters:**
- `max_generations` - Number of evolutionary generations to run
- `pop_size` - Number of solutions to maintain in population
- `max_sample_nums` - Maximum new solutions to sample per generation

**Parallel Execution:**
- `num_samplers` - Parallel LLM sampling workers
- `num_evaluators` - Parallel evaluation workers

**Logging:**
- `verbose` - Enable detailed progress logging

### Important Note

**LLM temperature and other sampling parameters are set when creating `HttpsApi`, NOT in algorithm configuration.**

```python
from evotoolkit.tools.llm import HttpsApi

# LLM configuration happens here
llm_api = HttpsApi(
    api_key="your-key",
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,  # LLM temperature
    max_tokens=4096
)

# Algorithm config does NOT include temperature
config = EvoEngineerConfig(
    running_llm=llm_api,  # Pass configured LLM
    # ... other params
)
```

---

## FunSearch Configuration

FunSearch uses an island model for continuous evolution.

```python
from evotoolkit.evo_method.funsearch import FunSearchConfig

config = FunSearchConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,

    # Sampling parameters
    max_sample_nums=30,           # Maximum number of samples
    programs_per_prompt=2,        # Programs generated per prompt

    # Island model
    num_islands=4,                # Number of parallel evolution islands
    max_population_size=1000,     # Maximum population size per island

    # Parallel control
    num_samplers=5,               # Number of parallel samplers
    num_evaluators=5,             # Number of parallel evaluators

    # Logging
    verbose=True
)
```

### Key Parameters

**Sampling:**
- `max_sample_nums` - Total samples to generate
- `programs_per_prompt` - Solutions per LLM call

**Island Model:**
- `num_islands` - Independent evolution islands (increases diversity)
- `max_population_size` - Maximum solutions per island

**Note:** FunSearch does **NOT** use `max_generations`. It evolves continuously based on the island model until `max_sample_nums` is reached.

### When to Use FunSearch

- When you want continuous evolution without fixed generations
- For exploring diverse solution spaces
- When you have computational resources for large populations

---

## EoH Configuration

EoH (Evolution of Heuristics) provides explicit control over genetic operators.

```python
from evotoolkit.evo_method.eoh import EoHConfig

config = EoHConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,

    # Evolution parameters
    max_generations=10,       # Maximum number of generations
    pop_size=5,               # Population size
    max_sample_nums=20,       # Maximum samples per generation
    selection_num=2,          # Number of parents for crossover

    # Operator control
    use_e2_operator=True,     # Use E2 operator (crossover)
    use_m1_operator=True,     # Use M1 operator (mutation)
    use_m2_operator=True,     # Use M2 operator (second mutation)

    # Parallel control
    num_samplers=5,           # Number of parallel samplers
    num_evaluators=5,         # Number of parallel evaluators

    # Logging
    verbose=True
)
```

### Key Parameters

**Evolution:**
- `max_generations` - Number of generations
- `pop_size` - Population size (typically smaller than EvoEngineer)
- `max_sample_nums` - Samples per generation
- `selection_num` - Parents selected for crossover

**Genetic Operators:**
- `use_e2_operator` - Enable/disable crossover operator
- `use_m1_operator` - Enable/disable first mutation operator
- `use_m2_operator` - Enable/disable second mutation operator

### When to Use EoH

- When you want explicit control over genetic operators
- For research comparing different operator combinations
- When traditional EA concepts are important

---

## Tuning Guidelines

### Population Size

**Small (5-10):**
- Pros: Faster generations, lower cost
- Cons: Less diversity, may converge too quickly
- Best for: Simple problems, limited resources

**Medium (10-20):**
- Pros: Good balance of speed and diversity
- Cons: None major
- Best for: Most problems (recommended default)

**Large (20+):**
- Pros: Maximum diversity, thorough exploration
- Cons: Slower, higher cost
- Best for: Complex problems, research

---

### Parallel Execution

```python
config = EvoEngineerConfig(
    # ... other params
    num_samplers=4,      # Parallel LLM calls
    num_evaluators=4,    # Parallel evaluations
)
```

**Guidelines:**
- `num_samplers`: Set based on LLM API rate limits
- `num_evaluators`: Set based on CPU/GPU availability
- Start conservatively (2-4) and increase if resources allow

**Example Configurations:**

```python
# Conservative (low resources)
num_samplers=2
num_evaluators=2

# Balanced (moderate resources)
num_samplers=4
num_evaluators=4

# Aggressive (high resources)
num_samplers=8
num_evaluators=8
```

---

### Generation Count

**Few Generations (5-10):**
- Quick experiments
- Simple problems
- Rapid prototyping

**Medium Generations (10-20):**
- Most problems
- Balanced exploration
- Recommended default

**Many Generations (20+):**
- Complex problems
- Research studies
- Final optimization runs

---

## Configuration Presets

### Quick Experimentation

```python
config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    pop_size=5,
    max_sample_nums=10,
    num_samplers=2,
    num_evaluators=2,
    verbose=True
)
```

**Use for:** Testing, debugging, rapid iteration

---

### Balanced Performance

```python
config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=15,
    pop_size=10,
    max_sample_nums=20,
    num_samplers=4,
    num_evaluators=4,
    verbose=True
)
```

**Use for:** Most production use cases

---

### Thorough Search

```python
config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=30,
    pop_size=15,
    max_sample_nums=30,
    num_samplers=6,
    num_evaluators=6,
    verbose=True
)
```

**Use for:** Research, benchmarking, final runs

---

## Next Steps

- Learn about [Algorithm Internals](internals.md) to analyze evolution behavior
- Check [Debugging & Profiling](debugging.md) for performance optimization
- Review the [API Reference](../../api/methods.md) for complete parameter details

---

## Resources

- [EvoEngineer Paper](https://arxiv.org/abs/...) - Algorithm details
- [FunSearch Paper](https://www.nature.com/articles/...) - Island model theory
- [EoH Paper](https://arxiv.org/abs/...) - Heuristic evolution