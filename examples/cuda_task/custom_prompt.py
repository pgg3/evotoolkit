# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Custom Prompt Example for CUDA Kernel Optimization

This example shows how to customize the evolution process by:
1. Inheriting from EvoEngineerFullCudaInterface
2. Overriding get_operator_prompt() to customize prompts for specific operators
3. Emphasizing memory optimization strategies in the prompt

Requirements:
- pip install evotoolkit[cuda_engineering]
- NVIDIA GPU with CUDA support
- Set LLM_API_URL and LLM_API_KEY environment variables
"""

import os
import tempfile
import evotoolkit
from evotoolkit.task.cuda_engineering import (
    CudaTask,
    CudaTaskInfoMaker,
    EvoEngineerFullCudaInterface,
)
from evotoolkit.task.cuda_engineering.evaluator import Evaluator
from evotoolkit.tools.llm import HttpsApi

# Set CUDA environment variables (required for CUDA kernel compilation)
# CUDA_HOME: Path to CUDA installation directory
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
# TORCH_CUDA_ARCH_LIST: GPU compute capability (e.g., "8.9" for RTX 4090)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")


class MemoryOptimizedCudaInterface(EvoEngineerFullCudaInterface):
    """Custom Interface optimized for memory-bound kernels.

    This interface customizes the mutation operator prompt to emphasize
    memory access patterns and optimization strategies.
    """

    def get_operator_prompt(
        self,
        operator_name,
        selected_individuals,
        current_best_sol,
        random_thoughts,
        **kwargs,
    ):
        """Customize prompt for mutation operator to emphasize memory optimization."""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            individual = selected_individuals[0]

            # Build thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join(
                    [f"- {thought}" for thought in random_thoughts]
                )
                thoughts_section = f"""
{thoughts_list}
"""

            prompt = f"""# CUDA KERNEL OPTIMIZATION - MEMORY FOCUS
{task_description}

## CURRENT BEST KERNEL
**Name:** {current_best_sol.other_info["name"]}
**Runtime:** {-current_best_sol.evaluation_res.score:.5f} milliseconds
**Approach:** {current_best_sol.other_info["thought"]}
**Performance Profile:**
{current_best_sol.evaluation_res.additional_info["prof_string"]}

## KERNEL TO MUTATE
**Name:** {individual.other_info["name"]}
**Runtime:** {-individual.evaluation_res.score:.5f} milliseconds
**Approach:** {individual.other_info["thought"]}
**Kernel Code:**
```cpp
{individual.sol_string}
```

## MEMORY OPTIMIZATION INSIGHTS
{thoughts_section}

## OPTIMIZATION STRATEGY
Focus on optimizing memory access patterns:

**Key Memory Optimizations:**
- **Shared Memory**: Use shared memory to reduce global memory accesses
- **Memory Coalescing**: Ensure memory accesses are coalesced for better bandwidth
- **Bank Conflicts**: Avoid shared memory bank conflicts
- **Memory Types**: Consider using texture memory or constant memory where appropriate
- **Prefetching**: Implement data prefetching to hide memory latency
- **Tiling**: Use tiling strategies to improve cache locality

{"Use the insights above if relevant as mutation guidance." if random_thoughts and len(random_thoughts) > 0 else ""}
Create a substantially modified version that reduces memory bottlenecks.

## RESPONSE FORMAT:
name: [descriptive_name_with_underscores]
code:
```cpp
[Your CUDA kernel implementation]
```
thought: [Memory optimization rationale and expected improvements]

## FORMAT REQUIREMENTS:
1. The code MUST be wrapped in ```cpp and ``` markers
2. The PYBIND11_MODULE inside the code has to be the same as in the CURRENT BEST KERNEL
3. MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE
4. Focus on memory access patterns and bandwidth optimization
"""
            return [{"role": "user", "content": prompt}]

        # Use default prompts for init and crossover operators
        return super().get_operator_prompt(
            operator_name,
            selected_individuals,
            current_best_sol,
            random_thoughts,
            **kwargs,
        )


def main():
    print("=" * 60)
    print("Custom Prompt CUDA Optimization Example")
    print("=" * 60)

    # Define Python reference implementation
    print("\n[1/4] Defining Python reference implementation...")

    org_py_code = '''
import torch

def matmul(A, B):
    """Matrix multiplication using PyTorch."""
    return torch.matmul(A, B)
'''

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

    # Define initial CUDA kernel
    cuda_code = """
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
"""

    # Create CUDA task
    print("\n[2/4] Creating CUDA optimization task...")

    temp_path = tempfile.mkdtemp()
    evaluator = Evaluator(temp_path)

    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=org_py_code,
        func_py_code=func_py_code,
        cuda_code=cuda_code,
        fake_mode=False,
    )

    task = CudaTask(data=task_info, temp_path=temp_path, fake_mode=False)
    print(f"GPU Type: {task.task_info['gpu_type']}")

    # Create custom interface
    print("\n[3/4] Setting up custom memory-optimized interface...")
    interface = MemoryOptimizedCudaInterface(task)
    print("Using custom prompt that emphasizes memory optimization strategies")

    # Configure LLM
    print("\n[4/4] Configuring LLM API...")
    # Set LLM_API_URL and LLM_API_KEY environment variables before running
    llm_api = HttpsApi(
        api_url=os.environ.get(
            "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
        ),
        key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
        model="gpt-4o",
    )

    # Run evolution
    print("\n" + "=" * 60)
    print("Starting evolution with custom prompts...")
    print("=" * 60 + "\n")

    result = evotoolkit.solve(
        interface=interface,
        output_path="./custom_prompt_results",
        running_llm=llm_api,
        max_generations=10,
        pop_size=5,
        max_sample_nums=20,
    )

    # Display results
    print("\n" + "=" * 60)
    print("Evolution completed!")
    print("=" * 60)
    print(f"\nOptimized runtime: {-result.evaluation_res.score:.4f} ms")
    print("Results saved to: ./custom_prompt_results/")
    print(f"\nBest kernel:\n{result.sol_string[:500]}...")


if __name__ == "__main__":
    main()
