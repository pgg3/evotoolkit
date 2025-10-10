# CUDA Kernel Optimization Tutorial

Learn how to optimize CUDA kernels using LLM-driven evolution to reduce runtime while maintaining correctness.

!!! note "Academic Citation"
    The CUDA kernel optimization task is based on EvoEngineer research. If you use this feature in academic work, please cite:

    ```bibtex
    @misc{guo2025evoengineermasteringautomatedcuda,
        title={EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models},
        author={Ping Guo and Chenyu Zhu and Siyuan Chen and Fei Liu and Xi Lin and Zhichao Lu and Qingfu Zhang},
        year={2025},
        eprint={2510.03760},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2510.03760}
    }
    ```

!!! tip "Complete Example Code"
    This tutorial provides complete, runnable examples (click to view/download):

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/basic_example.py) - Basic usage
    - [:material-download: dataset_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/dataset_example.py) - Using predefined dataset
    - [:material-download: custom_prompt.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/custom_prompt.py) - Custom prompt example
    - [:material-download: compare_algorithms.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/compare_algorithms.py) - Algorithm comparison
    - [:material-file-document: README.md](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/README.md) - Examples documentation and usage guide

    Run locally:
    ```bash
    cd examples/cuda_task
    python basic_example.py
    # or use predefined dataset
    python dataset_example.py
    ```

---

## Overview

This tutorial demonstrates:

- Creating CUDA kernel optimization tasks
- Optimizing kernel runtime using LLM-driven evolution
- Automatically verifying kernel correctness
- Evolving high-performance GPU code

---

## Installation

!!! tip "GPU Recommended"
    CUDA kernel optimization requires a GPU and PyTorch. Install PyTorch with CUDA support before EvoToolkit.
    We recommend **CUDA 12.9** (latest stable).

### Step 1: Install PyTorch with GPU Support

```bash
# CUDA 12.9 (recommended - for custom tasks)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# For other versions, visit: https://pytorch.org/get-started/locally/
# CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended for CUDA tasks)
# pip install torch torchvision
```

!!! note "About PyTorch Versions"
    We recommend installing the latest CUDA 12.9 version for custom task development. However, please note:

    - **Predefined datasets**: Our example datasets are built on **CUDA 12.4 + PyTorch 2.4.0**
    - **Version compatibility**: Different PyTorch versions may generate different CUDA code. When using predefined datasets, consider installing matching PyTorch versions
    - **Custom tasks**: If you're creating your own tasks, you can use any PyTorch version

### Step 2: Install EvoToolkit

```bash
pip install evotoolkit[cuda_engineering]
```

This installs:

- Ninja (high-performance build system)
- Portalocker (cross-process file locking)
- Psutil (system and process utilities)

### Step 3: Install C++ Compiler (Required)

!!! danger "Critical Prerequisite: C++ Compiler"
    **CUDA kernel compilation requires a C++ compiler!** Without it, you'll encounter errors like:
    ```
    Error checking compiler version for cl: [WinError 2] The system cannot find the file specified.
    ```

#### Windows Users

**You must install Visual Studio with MSVC compiler:**

1. **Download Visual Studio**
   - Visit: [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)
   - Recommended: **Visual Studio 2022 Community** (free)

2. **Select Workload During Installation**
   - Check **"Desktop development with C++"**
   - This installs MSVC compiler and necessary build tools

3. **CUDA Version & MSVC Compatibility**

   | CUDA Version | Supported Visual Studio | Supported MSVC |
   |-------------|------------------------|----------------|
   | 12.9        | VS 2022 (17.x)<br>VS 2019 (16.x) | MSVC 193x<br>MSVC 192x |
   | 12.4        | VS 2022 (17.x)<br>VS 2019 (16.x) | MSVC 193x<br>MSVC 192x |
   | 12.1        | VS 2022 (17.x)<br>VS 2019 (16.x)<br>VS 2017 (15.x) | MSVC 193x<br>MSVC 192x<br>MSVC 191x |

!!! warning "Important Notes"
    - Visual Studio 2017 deprecated in CUDA 12.5, completely removed in 12.9
    - Only 64-bit compilation supported from CUDA 12.0 onwards (no 32-bit)
    - Supports C++14 (default), C++17, and C++20

4. **Verify Compiler Installation**
   ```bash
   # Open "x64 Native Tools Command Prompt for VS 2022" (find it in Start menu)
   cl

   # Should see output like:
   # Microsoft (R) C/C++ Optimizing Compiler Version 19.39.xxxxx for x64
   ```

   If `cl` command is not available in regular Command Prompt, use one of these solutions:

   **Solution A: Use VS Developer Command Prompt (Recommended)**
   - Search for "x64 Native Tools Command Prompt for VS 2022" in Start menu
   - Run your Python scripts in this prompt

   **Solution B: Add to System PATH (Permanent)**
   ```bash
   # Add MSVC to system PATH environment variable (example path, adjust to your installation)
   # C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.xxxxx\bin\Hostx64\x64
   ```

#### Linux/Ubuntu Users

**Install GCC/G++ compiler:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential

# Verify installation
gcc --version
g++ --version

# Recommended: GCC 9.x or higher
```

**CUDA Version & GCC Compatibility:**

| CUDA Version | Supported GCC Versions |
|-------------|----------------------|
| 12.9        | GCC 9.x - 13.x |
| 12.4        | GCC 9.x - 13.x |
| 12.1        | GCC 9.x - 12.x |

!!! tip "Check CUDA & Compiler Compatibility"
    If you encounter compilation errors:

    1. Check CUDA version: `nvcc --version`
    2. Check compiler version: `cl` on Windows, `gcc --version` on Linux
    3. Verify versions are within compatibility ranges above

**Prerequisites Summary:**

- ✅ NVIDIA GPU with CUDA support
- ✅ CUDA toolkit installed (12.1+ recommended)
- ✅ **Compatible C++ compiler** (Windows: MSVC, Linux: GCC)
- ✅ PyTorch >= 2.0 (with CUDA support)
- ✅ Basic understanding of CUDA programming
- ✅ Familiarity with kernel optimization concepts

---

## Understanding CUDA Tasks

### What is a CUDA Task?

A CUDA task optimizes GPU kernel code to minimize runtime while ensuring correctness. The framework:

1. Takes your Python function implementation
2. Converts it to functional Python code (if needed)
3. Translates to initial CUDA kernel
4. Evolves the kernel to improve performance
5. Validates correctness against the Python reference

### Task Components

A CUDA task requires:

- **Original Python Code** (`org_py_code`): Original PyTorch model code (optional, can be empty)
- **Functional Python Code** (`func_py_code`): Extracted functional implementation for correctness comparison and performance benchmarking
- **CUDA Code** (`cuda_code`): Initial CUDA kernel implementation
- **GPU Info**: GPU type and CUDA version

!!! note "About org_py_code and func_py_code"
    - `func_py_code` must be provided - it's the actual Python reference used for CUDA correctness validation and performance comparison
    - If you only have `org_py_code`, you can use the AI-CUDA-Engineer workflow (Stage 0) to convert it to `func_py_code` using LLM
    - `org_py_code` can be empty if you provide `func_py_code` directly (recommended for evolution optimization)

---

!!! warning "Windows Users: Multiprocessing Protection Required"
    **CUDA task evaluator uses the multiprocessing module for timeout control. On Windows, you MUST protect all main code with `if __name__ == '__main__':` or it will cause infinite process recursion!**

    **Wrong example (causes RuntimeError):**
    ```python
    # ❌ Wrong - no protection
    import os
    from evotoolkit.task.cuda_engineering import CudaTask

    evaluator = Evaluator(temp_path)  # Will crash on Windows!
    task_info = CudaTaskInfoMaker.make_task_info(...)
    ```

    **Correct example:**
    ```python
    # ✅ Correct - use if __name__ == '__main__': protection
    import os
    from evotoolkit.task.cuda_engineering import CudaTask

    def main():
        evaluator = Evaluator(temp_path)
        task_info = CudaTaskInfoMaker.make_task_info(...)
        # ... other code

    if __name__ == '__main__':
        main()
    ```

    **Why is this protection needed?**

    - Windows doesn't support `fork`, only `spawn` for starting subprocesses
    - `spawn` re-imports the main module to create subprocesses
    - Without protection, every import re-executes main code, causing infinite recursion

    **Rule: Any code that calls CUDA task evaluation MUST be inside `if __name__ == '__main__':` protection!**

---

## Using Predefined Datasets

EvoToolkit provides predefined CUDA optimization datasets containing various common deep learning operations.

### Downloading the Dataset

The dataset is not included in the main repository and needs to be downloaded separately:

**Download methods:**

```bash
# Method 1: Using wget
cd /path/to/evotool/project/root
wget https://github.com/pgg3/evotoolkit/releases/download/data-v1.0.0/rtx4090_cu12_4_py311_torch_2_4_0.json

# Method 2: Using curl
curl -L -O https://github.com/pgg3/evotoolkit/releases/download/data-v1.0.0/rtx4090_cu12_4_py311_torch_2_4_0.json
```

**Dataset information:**

- **Filename:** `rtx4090_cu12_4_py311_torch_2_4_0.json`
- **Size:** ~580 KB
- **Format:** JSON
- **Optimized for:** RTX 4090 GPU + CUDA 12.4.1 + PyTorch 2.4.0

!!! warning "Dataset Note"
    This is a sample dataset for specific hardware/software configuration. Unlike scientific_regression tasks, it does not support automatic download. You can create similar datasets for your own hardware environment.

### Loading a Dataset

```python
import json

# Load dataset for RTX 4090 + CUDA 12.4.1 + PyTorch 2.4.0
with open('rtx4090_cu12_4_py311_torch_2_4_0.json', 'r') as f:
    dataset = json.load(f)

# View available tasks
print(f"Available tasks: {len(dataset)}")
print(f"Task list: {list(dataset.keys())[:5]}...")  # Show first 5

# Select a task
task_name = "10_3D_tensor_matrix_multiplication"
task_data = dataset[task_name]

print(f"\nTask: {task_name}")
print(f"- org_py_code: {'Provided' if task_data['org_py_code'] else 'Empty'}")
print(f"- func_py_code: {'Provided' if task_data['func_py_code'] else 'Empty'}")
print(f"- cuda_code: {'Provided' if task_data['cuda_code'] else 'Empty'}")
```

**Dataset includes task types:**

- Matrix multiplication variants (3D, 4D tensors, diagonal, symmetric matrices, etc.)
- Activation functions (ReLU, Sigmoid, Tanh, GELU, etc.)
- Loss functions (CrossEntropy, HingeLoss, etc.)
- Normalization layers (LayerNorm, BatchNorm, etc.)
- Attention mechanisms and Transformer components

### Creating a Task from Dataset

```python
from evotoolkit.task.cuda_engineering import CudaTask, CudaTaskInfoMaker
from evotoolkit.task.cuda_engineering.evaluator import Evaluator
import tempfile
import os


def main():
    # Configure CUDA environment variables (must be set before running)
    # Windows: Set to your CUDA installation path
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
    # Linux/Ubuntu: Usually the default path
    # os.environ["CUDA_HOME"] = "/usr/local/cuda"

    # Specify GPU architecture to save compilation time
    # RTX 4090: 8.9, RTX 3090: 8.6, V100: 7.0
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"

    # Use task data from dataset
    task_data = dataset["10_3D_tensor_matrix_multiplication"]

    # Create evaluator and task
    temp_path = tempfile.mkdtemp()
    evaluator = Evaluator(temp_path)

    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=task_data["org_py_code"],      # Can be empty
        func_py_code=task_data["func_py_code"],    # Functional implementation
        cuda_code=task_data["cuda_code"],          # Initial CUDA kernel
        fake_mode=False
    )

    task = CudaTask(data=task_info, temp_path=temp_path, fake_mode=False)
    print(f"Task created, initial runtime: {task.task_info['cuda_info']['runtime']:.4f} ms")


if __name__ == '__main__':
    main()
```

---

## Example: Creating Matrix Multiplication from Scratch

If you want to create your own CUDA optimization task from scratch:

### Step 1: Prepare Your Python Function

!!! note "func_py_code Format Requirements"
    `func_py_code` must contain the following components:

    1. **`module_fn` function**: Core functionality implementation
    2. **`Model` class**: Inherits from `nn.Module`, with `forward` method accepting `fn=module_fn` parameter
    3. **`get_inputs()` function**: Generates test input data
    4. **`get_init_inputs()` function**: Generates initialization inputs (usually empty list)

    This design allows CUDA kernels to replace `module_fn` by passing different `fn`, enabling correctness validation.

```python
# Original function to optimize (optional)
org_py_code = '''
import torch

def matmul(A, B):
    """Matrix multiplication using PyTorch."""
    return torch.matmul(A, B)
'''

# Functional implementation (for correctness comparison and benchmarking)
func_py_code = '''
import torch
import torch.nn as nn

def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Functional matrix multiplication implementation."""
    return torch.matmul(A, B)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)

M = 1024
K = 2048
N = 1024

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
'''
```

### Step 2: Create Initial CUDA Kernel

```python
# Initial CUDA implementation (naive version)
cuda_code = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(float* A, float* B, float* C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA)");
}
'''
```

### Step 3: Create CUDA Task

```python
from evotoolkit.task.cuda_engineering import CudaTask, CudaTaskInfoMaker
from evotoolkit.task.cuda_engineering.evaluator import Evaluator
import tempfile
import os


def main():
    # Configure CUDA environment variables (must be set before running)
    # Windows: Set to your CUDA installation path
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
    # Linux/Ubuntu: Usually the default path
    # os.environ["CUDA_HOME"] = "/usr/local/cuda"

    # Specify GPU architecture to save compilation time
    # RTX 4090: 8.9, RTX 3090: 8.6, V100: 7.0
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"

    # Create evaluator
    temp_path = tempfile.mkdtemp()
    evaluator = Evaluator(temp_path)

    # Create task info
    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=org_py_code,
        func_py_code=func_py_code,
        cuda_code=cuda_code,
        fake_mode=False  # Set True for testing without GPU
    )

    # Create task
    task = CudaTask(
        data=task_info,
        temp_path=temp_path,
        fake_mode=False
    )

    print(f"GPU Type: {task.task_info['gpu_type']}")
    print(f"CUDA Version: {task.task_info['cuda_version']}")
    print(f"Initial runtime: {task.task_info['cuda_info']['runtime']:.4f} ms")


if __name__ == '__main__':
    main()
```

**Output:**
```
GPU Type: RTX 4090
CUDA Version: 12.4.1
Initial runtime: 2.3456 ms
```

### Step 4: Test with Initial Solution

```python
def main():
    # ... (previous Step 3 code)

    # Get initial solution
    init_sol = task.make_init_sol_wo_other_info()

    print("Initial kernel info:")
    print(f"Runtime: {-init_sol.evaluation_res.score:.4f} ms")
    print(f"Score: {init_sol.evaluation_res.score:.6f}")


if __name__ == '__main__':
    main()
```

**Understanding Evaluation:**

- **Score**: Negative runtime (higher is better, so faster kernels have higher scores)
- **Runtime**: Kernel execution time in milliseconds
- **Correctness**: Automatically verified against Python reference
- **Profile String**: CUDA profiler output showing bottlenecks

### Step 5: Run Evolution with EvoEngineer

!!! note "Complete Code Example"
    The following code assumes you have completed the previous steps (Steps 1-4) and the `task` object has been created. For a complete runnable code example, please refer to [basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/basic_example.py).

```python
import os
import evotoolkit
from evotoolkit.task.cuda_engineering import EvoEngineerFullCudaInterface
from evotoolkit.tools.llm import HttpsApi


def main():
    # === Previous Steps (Steps 1-4) ===
    # This should include code from previous steps:
    # - Define org_py_code, func_py_code, cuda_code
    # - Create evaluator and task_info
    # - Create task object
    # See basic_example.py for complete code

    # Set CUDA environment variables (required for CUDA kernel compilation)
    # CUDA_HOME: Path to CUDA installation directory
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    # TORCH_CUDA_ARCH_LIST: GPU compute capability (e.g., "8.9" for RTX 4090)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    # Create interface (using the task object from previous steps)
    interface = EvoEngineerFullCudaInterface(task)

    # Configure LLM API
    # Set LLM_API_URL and LLM_API_KEY environment variables
    llm_api = HttpsApi(
        api_url=os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
        model="gpt-4o"
    )

    # Run evolution
    result = evotoolkit.solve(
        interface=interface,
        output_path='./cuda_optimization_results',
        running_llm=llm_api,
        max_generations=10,
        pop_size=5,
        max_sample_nums=20
    )

    print(f"Best kernel found!")
    print(f"Runtime: {-result.evaluation_res.score:.4f} ms")
    print(f"Speedup: {task.task_info['cuda_info']['runtime'] / (-result.evaluation_res.score):.2f}x")
    print(f"\nOptimized kernel:\n{result.sol_string}")


if __name__ == '__main__':
    main()
```

!!! tip "Try Other Algorithms"
    EvoToolkit supports multiple evolution algorithms for CUDA optimization:

    ```python
    # Use EoH
    from evotoolkit.task.cuda_engineering import EoHCudaInterface
    interface = EoHCudaInterface(task)

    # Use FunSearch
    from evotoolkit.task.cuda_engineering import FunSearchCudaInterface
    interface = FunSearchCudaInterface(task)

    # Use EvoEngineer with Insights
    from evotoolkit.task.cuda_engineering import EvoEngineerInsightCudaInterface
    interface = EvoEngineerInsightCudaInterface(task)

    # Use EvoEngineer Free-form
    from evotoolkit.task.cuda_engineering import EvoEngineerFreeCudaInterface
    interface = EvoEngineerFreeCudaInterface(task)
    ```

    Then use the same `evotoolkit.solve()` call to run evolution. Different interfaces may perform better for different kernels.

---

## Customizing Evolution Behavior

The quality of the evolutionary process is primarily controlled by the **evolution method** and its internal **prompt design**. If you want to improve results:

- **Adjust prompts**: Inherit existing Interface classes and customize LLM prompts
- **Develop new algorithms**: Create brand new evolutionary strategies and operators

!!! tip "Learn More"
    These are universal techniques applicable to all tasks. For detailed tutorials, see:

    - **[Customizing Evolution Methods](../customization/customizing-evolution.md)** - How to modify prompts and develop new algorithms
    - **[Advanced Usage](../advanced-overview.md)** - More advanced configuration options

**Quick Example - Customize prompt for CUDA optimization:**

````python
from evotoolkit.task.cuda_engineering import EvoEngineerFullCudaInterface

class OptimizedCudaInterface(EvoEngineerFullCudaInterface):
    """Interface optimized for memory-bound kernels."""

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """Customize mutation prompt to emphasize memory access patterns."""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            individual = selected_individuals[0]

            prompt = f"""# CUDA KERNEL OPTIMIZATION - MEMORY FOCUS
{task_description}

## CURRENT BEST
**Name:** {current_best_sol.other_info['name']}
**Runtime:** {-current_best_sol.evaluation_res.score:.5f} milliseconds

## KERNEL TO MUTATE
**Name:** {individual.other_info['name']}
**Runtime:** {-individual.evaluation_res.score:.5f} milliseconds

## OPTIMIZATION FOCUS
Focus on optimizing memory access patterns:
- Use shared memory to reduce global memory accesses
- Implement memory coalescing for better bandwidth
- Consider memory bank conflicts
- Use appropriate memory access patterns (texture, constant memory)

Generate an improved kernel that reduces memory bottlenecks.

## RESPONSE FORMAT:
name: [descriptive_name]
code:
```cpp
[Your CUDA kernel implementation]
```
thought: [Memory optimization rationale]
"""
            return [{"role": "user", "content": prompt}]

        # Use default prompts for other operators
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# Use custom interface
interface = OptimizedCudaInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10
)
````

!!! note "About EvoEngineer Operators"
    EvoEngineer uses three operators: **init** (initialization), **mutation** (mutation), **crossover** (crossover).
    The parent class `EvoEngineerFullCudaInterface` already defines these operators and default prompts.
    You only need to override `get_operator_prompt()` to customize specific operator prompts - others will automatically use the default implementation.

For complete customization tutorials and more examples, see [Customizing Evolution Methods](../customization/customizing-evolution.md).

---

## Understanding Evaluation

### How Scoring Works

1. **Correctness Validation**: CUDA kernel output is compared against Python reference implementation
2. **Runtime Measurement**: Kernel execution time is measured using CUDA events and profiling
3. **Fitness**: Negative runtime (higher is better, so lower runtime = higher fitness)

### Evaluation Output

```python
result = task.evaluate_code(candidate_cuda_code)

if result.valid:
    print(f"Score: {result.score}")                                    # Higher is better
    print(f"Runtime: {-result.score:.4f} ms")                          # Actual runtime
    print(f"Profile: {result.additional_info['prof_string']}")         # CUDA profiler output
else:
    if result.additional_info['compilation_error']:
        print(f"Compilation error: {result.additional_info['error_msg']}")
    elif result.additional_info['comparison_error']:
        print(f"Correctness error: {result.additional_info['error_msg']}")
```

### Fake Mode for Testing

You can test without GPU using fake mode:

```python
def main():
    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=org_py_code,
        func_py_code=func_py_code,
        cuda_code=cuda_code,
        fake_mode=True  # Skip actual CUDA evaluation
    )

    task = CudaTask(data=task_info, fake_mode=True)


if __name__ == '__main__':
    main()
```

---

## FAQ

### Q: How to handle the `_get_vc_env is private` warning?

**Problem Description:**

When compiling CUDA extensions on Windows, you may see the following warning:

```
UserWarning: _get_vc_env is private; find an alternative (pypa/distutils#340)
```

**Root Cause:**

This is a compatibility warning from setuptools/distutils when detecting the MSVC compiler on Windows:

- CUDA extension compilation requires Visual Studio C++ compiler (MSVC)
- setuptools calls the internal function `_get_vc_env()` to get compiler environment
- Python is migrating distutils from stdlib to setuptools, and some internal APIs are marked as private during this transition

**Impact:**

- ⚠️ This is just a UserWarning, **it does not affect program execution**
- ✅ **Does not affect CUDA kernel compilation**
- ✅ **Does not affect optimization results**

**Solutions:**

**Solution 1: Filter the warning (Recommended)**

Add warning filter at the beginning of your script:

```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='setuptools')

# Or more precisely
warnings.filterwarnings('ignore', message='.*_get_vc_env is private.*')

# Then import other modules
from evotoolkit.task.cuda_engineering import CudaTask
# ...
```

**Solution 2: Upgrade setuptools**

Try upgrading to the latest version (may have fixed the issue):

```bash
pip install --upgrade setuptools
```

**Solution 3: Ignore it**

If you don't mind seeing the warning, you can simply ignore it. This warning doesn't affect functionality, it just reminds developers that the internal API may change in future versions.

---

### Q: Why is `if __name__ == '__main__':` protection required on Windows?

**Reason:**

- Windows does not support `fork` process creation, only `spawn`
- `spawn` re-imports the main module to create subprocesses
- CUDA task evaluator uses `multiprocessing` module for timeout control
- Without protection, every import will execute the main code, causing infinite recursive process creation

**Correct Example:**

```python
from evotoolkit.task.cuda_engineering import CudaTask

def main():
    evaluator = Evaluator(temp_path)
    task = CudaTask(...)
    # All task code

if __name__ == '__main__':
    main()
```

**Incorrect Example (will crash):**

```python
from evotoolkit.task.cuda_engineering import CudaTask

# ❌ Executing directly at module level
evaluator = Evaluator(temp_path)  # Will cause RuntimeError
```

---

## Next Steps

### Explore different optimization strategies

- Try different evolution algorithms (EvoEngineer variants, EoH, FunSearch)
- Compare results across different interfaces
- Analyze performance profiles to identify bottlenecks
- Experiment with different kernel patterns (tiled, shared memory, etc.)

### Customize and improve the evolution process

- Inspect prompt designs in existing Interface classes
- Inherit and override Interface to customize prompts
- Design specialized prompts for different optimization goals (memory-bound, compute-bound, etc.)
- If needed, develop brand new evolution algorithms

### Learn more

- [Customizing Evolution Methods](../customization/customizing-evolution.md) - Deep dive into prompt customization and algorithm development
- [Advanced Usage](../advanced-overview.md) - Advanced configurations and techniques
- [API Reference](../../api/index.md) - Complete API documentation
- [Development Docs](../../development/contributing.md) - Contributing new methods and features
