# Scientific Regression Examples

English | [简体中文](README.zh.md)

This directory contains complete, runnable examples for scientific symbolic regression tasks.

## Installation

```bash
pip install evotoolkit[scientific_regression]
```

## Setup

Set environment variable or configure API key directly in code:

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

Or set directly in code (see individual example files).

## Examples

### 1. Basic Example (`basic_example.py`)

**What it does:**
- Loads the bacterial growth dataset
- Creates a scientific regression task
- Runs evolution with EvoEngineer algorithm
- Discovers mathematical equations from data

**Run:**
```bash
python basic_example.py
```

**Expected runtime:** 3-5 minutes (3 generations)

---

### 2. Custom Prompt Example (`custom_prompt.py`)

**What it does:**
- Demonstrates how to customize evolution prompts
- Inherits from `EvoEngineerPythonInterface`
- Overrides mutation prompt to emphasize scientific principles
- Shows how to guide LLM toward biologically plausible equations

**Run:**
```bash
python custom_prompt.py
```

**Key learning:**
- How to create custom interfaces
- How to override `get_operator_prompt()`
- How to inject domain knowledge into prompts

---

### 3. Compare Algorithms (`compare_algorithms.py`)

**What it does:**
- Compares three evolution algorithms: EvoEngineer, EoH, FunSearch
- Runs all algorithms on the same task
- Ranks results by performance
- Shows which algorithm works best for this task

**Run:**
```bash
python compare_algorithms.py
```

**Expected runtime:** 10-15 minutes (3 algorithms × 3 generations each)

## Available Datasets

- `bactgrow`: E. coli bacterial growth rate (4 inputs: population, substrate, temp, pH)
- `oscillator1`: Damped nonlinear oscillator acceleration (2 inputs: position, velocity)
- `oscillator2`: Damped nonlinear oscillator variant (2 inputs: position, velocity)
- `stressstrain`: Aluminum rod stress prediction (2 inputs: strain, temperature)

## Output

All examples save results to their respective output directories:
- `basic_example.py` → `./scientific_regression_results/`
- `custom_prompt.py` → `./custom_prompt_results/`
- `compare_algorithms.py` → `./results_evoengineer/`, `./results_eoh/`, `./results_funsearch/`

Each directory contains:
- `run_state.json` - Evolution statistics
- `best_solution.py` - Best evolved equation
- `generation_N/` - Solutions from each generation

## Next Steps

- Try different datasets by changing `dataset_name` parameter
- Adjust evolution parameters (`max_generations`, `pop_size`, etc.)
- Customize prompts for your specific domain
- Develop new evolution algorithms

## Resources

For complete tutorials and API documentation, visit the EvoToolkit online documentation site.
