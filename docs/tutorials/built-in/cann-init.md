# CANN Init (Ascend NPU Operator) Tutorial

Learn how to use LLM-driven evolution to generate and optimize Ascend C operator kernel code for Huawei Ascend NPUs.

!!! warning "Hardware Requirement"
    This task requires **Huawei Ascend NPU hardware** and the **CANN toolkit** to be installed. It cannot run in standard CPU/GPU environments.

!!! note "Experimental Adjacent Workflow"
    CANN Init is kept in EvoToolkit as an experimental adjacent workflow. It is documented for completeness, but it is not part of the primary reviewed surface for the MLOSS submission.

!!! tip "Complete Example Code"
    See the example directory for scripts:

    - [:material-folder: examples/cann_init/](https://github.com/pgg3/evotoolkit/blob/master/examples/cann_init/) - Agent and evaluator scripts
    - [:material-file-document: README.md](https://github.com/pgg3/evotoolkit/blob/master/examples/cann_init/README.md) - Usage guide

---

## Overview

This tutorial demonstrates:

- Creating a CANN Init task for Ascend C operator generation
- Using LLM-driven evolution to generate optimized Ascend C kernel code
- Understanding the operator signature and template system
- Evaluating operator correctness and performance on Ascend NPU hardware

EvoToolkit treats Ascend C operator generation as an optimization problem: given a Python reference implementation, evolve Ascend C kernel code that is both correct and performant.

---

## Prerequisites

### Hardware

- Huawei Ascend NPU (tested on Ascend910B2)

### Software

```bash
# Install CANN toolkit from Huawei (see official documentation)
# https://www.hiascend.com/software/cann

# Install EvoToolkit with CANN support
pip install evotoolkit[cann_init]
```

This installs:

- `pybind11` - For Python/C++ binding generation
- Other CANN-related dependencies

---

## Understanding the CANN Init Task

### What Does the Task Generate?

The task evolves **Ascend C kernel code** (C++ for Ascend NPU). Given:

1. An **operator name** (e.g., `"relu"`, `"layer_norm"`)
2. A **Python reference implementation** (correct but not optimized)

The LLM generates Ascend C kernel code that implements the same operation using Ascend C APIs (Data Copy, Compute, Tiling, etc.).

### Template System

EvoToolkit automatically generates the surrounding code (host code, tiling configuration, Python bindings) from templates. The LLM only needs to provide the **kernel implementation**.

### Evaluation

Each generated kernel is:

1. **Compiled** using the CANN toolkit
2. **Tested for correctness** against the Python reference
3. **Benchmarked** for performance (throughput, latency)

---

## Quick Start

### Step 1: Define the Python Reference

```python
PYTHON_REFERENCE = '''
def relu(x):
    """ReLU activation: max(0, x)"""
    import numpy as np
    return np.maximum(0, x)
'''
```

### Step 2: Create the Task

```python
from evotoolkit.task.cann_init import CANNInitTask

task = CANNInitTask(
    data={
        "op_name": "relu",
        "python_reference": PYTHON_REFERENCE,
        "npu_type": "Ascend910B2",   # Your NPU model
        "cann_version": "8.0",        # Your CANN version
    },
    project_path="/tmp/cann_projects",  # Directory for compiled artifacts
)

print(f"Operator: {task.task_info['op_name']}")
print(f"NPU type: {task.task_info['npu_type']}")
```

### Step 3: Evaluate a Kernel

```python
kernel_code = '''
// Ascend C ReLU kernel implementation
class KernelRelu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        // ... initialization code ...
    }

    __aicore__ inline void Process() {
        // ... computation code ...
    }
};
'''

result = task.evaluate_code(kernel_code)

if result.valid:
    print(f"Score: {result.score:.4f}")
    print(f"Correctness: {result.additional_info.get('correctness')}")
    print(f"Performance: {result.additional_info.get('performance')}")
else:
    print(f"Error: {result.additional_info.get('error')}")
```

### Step 4: Run Evolution

```python
import evotoolkit
from evotoolkit.task.cann_init.method_interface import CANNIniterInterface
from evotoolkit.tools.llm import HttpsApi

# Create interface
interface = CANNIniterInterface(task)

# Configure LLM
llm_api = HttpsApi(
    api_url="api.openai.com",
    key="your-api-key-here",
    model="gpt-4o"
)

# Run evolution
result = evotoolkit.solve(
    interface=interface,
    output_path='./cann_results',
    running_llm=llm_api,
    max_generations=5,
    pop_size=3,
)

print(f"Best kernel found:")
print(result.sol_string)
print(f"Score: {result.evaluation_res.score:.4f}")
```

---

## `CANNInitTask` API

```python
class CANNInitTask(BaseTask):
    def __init__(
        self,
        data: dict,            # Task configuration (see below)
        project_path: str | None = None,  # Default directory for compiled artifacts
        fake_mode: bool = False,          # Skip evaluation (for testing)
    )
```

**`data` dictionary keys:**

| Key | Required | Description |
|-----|----------|-------------|
| `op_name` | Yes | Operator name (e.g., `"relu"`, `"layer_norm"`) |
| `python_reference` | Yes | Python reference implementation (string) |
| `npu_type` | No | NPU model (default: `"Ascend910B2"`) |
| `cann_version` | No | CANN version (default: `"8.0"`) |

**Key Methods:**

| Method | Description |
|--------|-------------|
| `evaluate_code(kernel_src)` | Evaluate kernel code string, returns `EvaluationResult` |
| `evaluate_solution(solution)` | Rich interface with `other_info` for advanced options |

**Advanced `evaluate_solution` options via `other_info`:**

```python
from evotoolkit.core import Solution

# Compile-only mode (for parallel workflows)
solution = Solution(
    sol_string=kernel_src,
    other_info={
        "project_path": "/compile/sol_001",
        "compile_only": True,
        "save_compile_to": "/compile/sol_001",
    }
)
compile_result = task.evaluate_solution(solution)

# Load pre-compiled artifact for testing
solution = Solution(
    sol_string="",
    other_info={
        "load_from": "/compile/sol_001",
    }
)
test_result = task.evaluate_solution(solution)
```

---

## Supported Operators

The CANN Init task can be applied to any operator expressible in Python:

| Category | Examples |
|----------|---------|
| Element-wise | ReLU, Sigmoid, GELU, Add, Multiply |
| Reduction | Softmax, LayerNorm, Sum, Mean |
| Matmul | GEMM, Attention (SDPA) |
| Custom | Any operator with a Python reference |

---

## Tips for Better Results

1. **Provide a clear Python reference** — The LLM uses it to understand the operator semantics
2. **Start with simple operators** (element-wise) before complex ones (matmul)
3. **Use `fake_mode=True`** during development to test the pipeline without hardware
4. **Check CANN documentation** for available Ascend C APIs and tiling patterns

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| Compilation error | Check CANN environment variables and toolkit installation |
| Correctness failure | Review the Python reference for edge cases |
| Performance below baseline | LLM may need domain knowledge about Ascend C tiling |

---

## Next Steps

- [Customizing Evolution Methods](../customization/customizing-evolution.md) — Add domain knowledge to prompts
- [Advanced Usage](../advanced-overview.md) — Parallel compilation and advanced workflows
- [API Reference](../../api/index.md) — Complete API documentation
