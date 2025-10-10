# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Dataset Example for CUDA Kernel Optimization

This example shows how to use the predefined CUDA optimization dataset
containing various deep learning operations.

Requirements:
- pip install evotoolkit[cuda_engineering]
- NVIDIA GPU with CUDA support
- Dataset file: rtx4090_cu12_4_py311_torch_2_4_0.json
- Set LLM_API_URL and LLM_API_KEY environment variables
"""

import os
import json
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


def main():
    print("=" * 60)
    print("CUDA Dataset Example")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1/5] Loading predefined CUDA dataset...")
    dataset_path = "../../../rtx4090_cu12_4_py311_torch_2_4_0.json"

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} tasks from dataset")

    # Show first 10 tasks
    task_names = list(dataset.keys())[:10]
    print("\nFirst 10 tasks:")
    for i, name in enumerate(task_names, 1):
        print(f"  {i}. {name}")

    # Step 2: Select a task
    print("\n[2/5] Selecting task: 10_3D_tensor_matrix_multiplication...")
    task_name = "10_3D_tensor_matrix_multiplication"
    task_data = dataset[task_name]

    print("\nTask details:")
    print(f"- org_py_code: {'Provided' if task_data['org_py_code'] else 'Empty'}")
    print(f"- func_py_code: {'Provided' if task_data['func_py_code'] else 'Empty'}")
    print(f"- cuda_code: {'Provided' if task_data['cuda_code'] else 'Empty'}")

    # Step 3: Create CUDA task from dataset
    print("\n[3/5] Creating CUDA task from dataset...")

    temp_path = tempfile.mkdtemp()
    evaluator = Evaluator(temp_path)

    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=task_data["org_py_code"],
        func_py_code=task_data["func_py_code"],
        cuda_code=task_data["cuda_code"],
        fake_mode=False,  # Set True for testing without GPU
    )

    task = CudaTask(data=task_info, temp_path=temp_path, fake_mode=False)

    print(f"GPU Type: {task.task_info['gpu_type']}")
    print(f"CUDA Version: {task.task_info['cuda_version']}")
    print(f"Initial runtime: {task.task_info['cuda_info']['runtime']:.4f} ms")

    # Step 4: Setup evolution
    print("\n[4/5] Setting up evolution...")
    interface = EvoEngineerFullCudaInterface(task)

    # Configure LLM API
    # Set LLM_API_URL and LLM_API_KEY environment variables before running
    llm_api = HttpsApi(
        api_url=os.environ.get(
            "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
        ),
        key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
        model="gpt-4o",
    )

    # Step 5: Run evolution
    print("\n[5/5] Starting evolution...")
    print("This may take several minutes...")
    print("=" * 60 + "\n")

    result = evotoolkit.solve(
        interface=interface,
        output_path=f"./dataset_results_{task_name}",
        running_llm=llm_api,
        max_generations=10,
        pop_size=5,
        max_sample_nums=20,
    )

    # Display results
    print("\n" + "=" * 60)
    print("Evolution completed!")
    print("=" * 60)
    print(f"\nTask: {task_name}")
    print(f"Initial runtime: {task.task_info['cuda_info']['runtime']:.4f} ms")
    print(f"Optimized runtime: {-result.evaluation_res.score:.4f} ms")
    speedup = task.task_info["cuda_info"]["runtime"] / (-result.evaluation_res.score)
    print(f"Speedup: {speedup:.2f}x")
    print(f"\nResults saved to: ./dataset_results_{task_name}/")


if __name__ == "__main__":
    main()
