# CUDA Kernel Optimization Examples

English | [简体中文](README.zh.md)

This directory contains complete, runnable examples for CUDA kernel optimization tasks using LLM-driven evolution.

## Installation

```bash
pip install evotoolkit[cuda_engineering]
```

**Prerequisites:**

- NVIDIA GPU with CUDA support
- CUDA toolkit installed (11.0+)
- PyTorch with CUDA support

## Setup

Create a `.env` file with your API credentials:

```bash
HOST=api.openai.com
KEY=sk-your-api-key-here
```

## Examples

### 1. Basic Example (`basic_example.py`)

**What it does:**

- Defines a Python reference implementation for matrix multiplication
- Creates an initial naive CUDA kernel
- Runs evolution with EvoEngineerFull algorithm
- Optimizes the kernel to reduce runtime while maintaining correctness

**Run:**

```bash
python basic_example.py
```

**Expected runtime:** 10-20 minutes (10 generations)

---

### 2. Dataset Example (`dataset_example.py`)

**What it does:**

- Loads predefined CUDA optimization dataset (RTX 4090 + CUDA 12.4.1 + PyTorch 2.4.0)
- Selects a 3D tensor matrix multiplication task
- Creates CUDA task from dataset
- Runs evolution to optimize the kernel

**Run:**

```bash
python dataset_example.py
```

**Dataset download:**

The dataset is not included in the repository due to its large size. You can download it from:

- **GitHub Release:** [Download rtx4090_cu12_4_py311_torch_2_4_0.json](https://github.com/pgg3/evotoolkit/releases/download/v0.1.0/rtx4090_cu12_4_py311_torch_2_4_0.json)
- **Size:** ~580 KB (JSON format)
- **Save to:** `../../../rtx4090_cu12_4_py311_torch_2_4_0.json` (project root directory)

Or use `wget`:
```bash
cd ../../../  # Go to project root
wget https://github.com/pgg3/evotoolkit/releases/download/v0.1.0/rtx4090_cu12_4_py311_torch_2_4_0.json
```

**Dataset includes:**

- 100+ predefined CUDA optimization tasks
- Matrix operations, activation functions, loss functions
- Normalization layers, attention mechanisms
- Complete with org_py_code, func_py_code, and cuda_code
- Optimized for RTX 4090 + CUDA 12.4.1 + PyTorch 2.4.0

**Expected runtime:** 10-20 minutes (10 generations)

---

### 3. Custom Prompt Example (`custom_prompt.py`)

**What it does:**

- Demonstrates how to customize evolution prompts
- Inherits from `EvoEngineerFullCudaInterface`
- Overrides mutation prompt to emphasize memory optimization strategies
- Shows how to guide LLM toward specific optimization goals

**Run:**

```bash
python custom_prompt.py
```

**Key learning:**

- How to create custom CUDA interfaces
- How to override `get_operator_prompt()`
- How to inject domain-specific optimization strategies into prompts

---

### 4. Compare Algorithms (`compare_algorithms.py`)

**What it does:**

- Compares five evolution algorithms on the same CUDA task:
  - EvoEngineerFull: Full workflow with init, mutation, crossover
  - EvoEngineerFree: Free-form optimization
  - EvoEngineerInsight: Insight-guided optimization
  - EoH: Evolution of Heuristics
  - FunSearch: Function search optimization
- Runs all algorithms on the same task
- Ranks results by runtime performance
- Shows which algorithm achieves the best speedup

**Run:**

```bash
python compare_algorithms.py
```

**Expected runtime:** 40-80 minutes (5 algorithms × 10 generations each)

## Available Interfaces

The examples demonstrate different CUDA optimization interfaces:

- **EvoEngineerFullCudaInterface**: Complete evolution with init, mutation, and crossover operators
- **EvoEngineerFreeCudaInterface**: Free-form optimization approach
- **EvoEngineerInsightCudaInterface**: Insight-guided optimization with performance analysis
- **EoHCudaInterface**: Evolution of Heuristics for CUDA
- **FunSearchCudaInterface**: Function search for GPU kernels

## Output

All examples save results to their respective output directories:

- `basic_example.py` → `./cuda_optimization_results/`
- `custom_prompt.py` → `./custom_prompt_results/`
- `compare_algorithms.py` → `./results_evoengineer_full/`, `./results_evoengineer_free/`, etc.

Each directory contains:

- `run_state.json` - Evolution statistics and history
- `best_solution.cu` - Best evolved CUDA kernel
- `generation_N/` - Solutions from each generation

## Understanding CUDA Task Evaluation

### Correctness Validation

All evolved kernels are automatically validated against the Python reference implementation to ensure correctness.

### Performance Measurement

- **Runtime**: Kernel execution time measured using CUDA events
- **Score**: Negative runtime (higher score = faster kernel)
- **Profile**: CUDA profiler output showing performance bottlenecks

### Fake Mode for Testing

You can test the evolution workflow without a GPU by setting `fake_mode=True`:

```python
task_info = CudaTaskInfoMaker.make_task_info(
    evaluator=evaluator,
    gpu_type="RTX 4090",
    cuda_version="12.4.1",
    org_py_code=org_py_code,
    func_py_code=func_py_code,
    cuda_code=cuda_code,
    fake_mode=True  # Skip actual CUDA evaluation
)
```

## Next Steps

- Try different CUDA operations (convolution, reduction, scan, etc.)
- Adjust evolution parameters (`max_generations`, `pop_size`, `max_sample_nums`)
- Customize prompts for specific optimization goals (memory-bound, compute-bound)
- Develop new evolution algorithms tailored to GPU optimization

## Documentation

For detailed tutorials, see:

- [CUDA Engineering Tutorial](../../docs/tutorials/cuda-engineering.md)
- [Custom Evolution Methods](../../docs/tutorials/customizing-evolution.md)
- [API Reference](../../docs/api/index.md)
