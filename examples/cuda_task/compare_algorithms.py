# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Compare Different Evolution Algorithms for CUDA Optimization

This example compares different evolutionary algorithms on the same CUDA task:
- EvoEngineerFull: Full CUDA engineering workflow with init, mutation, crossover
- EvoEngineerFree: Free-form CUDA optimization
- EvoEngineerInsight: Insight-guided CUDA optimization
- EoH: Evolution of Heuristics
- FunSearch: Function search optimization

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
    EvoEngineerFreeCudaInterface,
    EvoEngineerInsightCudaInterface,
    EoHCudaInterface,
    FunSearchCudaInterface
)
from evotoolkit.task.cuda_engineering.evaluator import Evaluator
from evotoolkit.tools.llm import HttpsApi

# Set CUDA environment variables (required for CUDA kernel compilation)
# CUDA_HOME: Path to CUDA installation directory
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
# TORCH_CUDA_ARCH_LIST: GPU compute capability (e.g., "8.9" for RTX 4090)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")


def create_task():
    """Create a shared CUDA task for all algorithms."""
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

    temp_path = tempfile.mkdtemp()
    evaluator = Evaluator(temp_path)

    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=org_py_code,
        func_py_code=func_py_code,
        cuda_code=cuda_code,
        fake_mode=False
    )

    return CudaTask(data=task_info, temp_path=temp_path, fake_mode=False)


def run_algorithm(algorithm_name, interface_class, task, llm_api, output_dir):
    """Run a single evolutionary algorithm and return results."""
    print(f"\n{'=' * 60}")
    print(f"Running {algorithm_name}...")
    print(f"{'=' * 60}\n")

    # Create interface
    interface = interface_class(task)

    # Run evolution
    result = evotoolkit.solve(
        interface=interface,
        output_path=output_dir,
        running_llm=llm_api,
        max_generations=10,
        pop_size=5,
        max_sample_nums=20
    )

    print(f"\n{algorithm_name} completed!")
    print(f"Best runtime: {-result.evaluation_res.score:.4f} ms")

    return {
        'algorithm': algorithm_name,
        'runtime': -result.evaluation_res.score,
        'score': result.evaluation_res.score,
        'solution': result.sol_string,
        'output_path': output_dir
    }


def main():
    print("=" * 60)
    print("Comparing Evolution Algorithms for CUDA Optimization")
    print("=" * 60)

    # Create shared task
    print("\n[1/2] Creating CUDA optimization task...")
    task = create_task()
    print(f"GPU Type: {task.task_info['gpu_type']}")
    print(f"CUDA Version: {task.task_info['cuda_version']}")
    print(f"Initial runtime: {task.task_info['cuda_info']['runtime']:.4f} ms")

    # Configure LLM (shared by all algorithms)
    print("\n[2/2] Configuring LLM API...")
    # Set LLM_API_URL and LLM_API_KEY environment variables before running
    llm_api = HttpsApi(
        api_url=os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
        model="gpt-4o"
    )

    # Define algorithms to compare
    algorithms = [
        ("EvoEngineerFull", EvoEngineerFullCudaInterface, "./results_evoengineer_full"),
        ("EvoEngineerFree", EvoEngineerFreeCudaInterface, "./results_evoengineer_free"),
        ("EvoEngineerInsight", EvoEngineerInsightCudaInterface, "./results_evoengineer_insight"),
        ("EoH", EoHCudaInterface, "./results_eoh"),
        ("FunSearch", FunSearchCudaInterface, "./results_funsearch")
    ]

    # Run all algorithms
    results = []
    initial_runtime = task.task_info['cuda_info']['runtime']

    for algo_name, interface_class, output_dir in algorithms:
        result = run_algorithm(algo_name, interface_class, task, llm_api, output_dir)
        result['speedup'] = initial_runtime / result['runtime']
        results.append(result)

    # Display comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    # Sort by runtime (lower is better)
    results.sort(key=lambda x: x['runtime'])

    print(f"\nInitial runtime: {initial_runtime:.4f} ms\n")
    print("Ranking (lower runtime is better):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['algorithm']}")
        print(f"   Runtime: {result['runtime']:.4f} ms")
        print(f"   Speedup: {result['speedup']:.2f}x")
        print(f"   Output: {result['output_path']}")

    print("\n" + "=" * 60)
    print(f"Winner: {results[0]['algorithm']}")
    print(f"Best runtime: {results[0]['runtime']:.4f} ms")
    print(f"Best speedup: {results[0]['speedup']:.2f}x")
    print("=" * 60)

    print(f"\nBest kernel from {results[0]['algorithm']}:")
    print(results[0]['solution'][:500] + "...")


if __name__ == "__main__":
    main()
