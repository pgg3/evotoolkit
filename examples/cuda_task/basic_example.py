# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Basic CUDA Kernel Optimization Example

This example demonstrates how to use EvoToolkit to optimize CUDA kernels for
matrix multiplication using LLM-driven evolution.

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
    EvoEngineerFullCudaInterface
)
from evotoolkit.task.cuda_engineering.evaluator import Evaluator
from evotoolkit.tools.llm import HttpsApi

# Set CUDA environment variables (required for CUDA kernel compilation)
# CUDA_HOME: Path to CUDA installation directory
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
# TORCH_CUDA_ARCH_LIST: GPU compute capability (e.g., "8.9" for RTX 4090)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")


def main():
    print("=" * 60)
    print("CUDA Kernel Optimization Example")
    print("=" * 60)

    # Step 1: Define Python reference implementation
    print("\n[1/5] Defining Python reference implementation...")

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

    # Step 2: Define initial CUDA kernel
    print("\n[2/5] Defining initial CUDA kernel...")

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

    # Step 3: Create CUDA task
    print("\n[3/5] Creating CUDA optimization task...")

    temp_path = tempfile.mkdtemp()
    evaluator = Evaluator(temp_path)

    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=org_py_code,
        func_py_code=func_py_code,
        cuda_code=cuda_code,
        fake_mode=False  # Set True for testing without GPU
    )

    task = CudaTask(
        data=task_info,
        temp_path=temp_path,
        fake_mode=False
    )

    print(f"GPU Type: {task.task_info['gpu_type']}")
    print(f"CUDA Version: {task.task_info['cuda_version']}")

    # Step 4: Test initial solution
    print("\n[4/5] Testing initial CUDA kernel...")
    init_sol = task.make_init_sol_wo_other_info()
    print(f"Initial kernel runtime: {-init_sol.evaluation_res.score:.4f} ms")
    print(f"Initial kernel score: {init_sol.evaluation_res.score:.6f}")

    # Step 5: Create interface and configure LLM
    print("\n[5/5] Setting up evolution...")
    interface = EvoEngineerFullCudaInterface(task)

    # Configure LLM API
    # Set LLM_API_URL and LLM_API_KEY environment variables before running
    llm_api = HttpsApi(
        api_url=os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
        model="gpt-4o"
    )

    # Run evolution
    print("\n" + "=" * 60)
    print("Starting CUDA kernel evolution...")
    print("This may take several minutes...")
    print("=" * 60 + "\n")

    result = evotoolkit.solve(
        interface=interface,
        output_path='./cuda_optimization_results',
        running_llm=llm_api,
        max_generations=10,
        pop_size=5,
        max_sample_nums=20
    )

    # Display results
    print("\n" + "=" * 60)
    print("Evolution completed!")
    print("=" * 60)
    print(f"\nInitial runtime: {-init_sol.evaluation_res.score:.4f} ms")
    print(f"Optimized runtime: {-result.evaluation_res.score:.4f} ms")
    print(f"Speedup: {(-init_sol.evaluation_res.score) / (-result.evaluation_res.score):.2f}x")
    print(f"\nResults saved to: ./cuda_optimization_results/")
    print(f"\nBest kernel:\n{result.sol_string[:500]}...")


if __name__ == "__main__":
    main()
