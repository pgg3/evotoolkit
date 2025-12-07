# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Basic Evaluation

Demonstrates CANNInitTask evaluation in two modes:
1. Default mode: Only kernel_src (for element-wise operators)
2. Full LLM mode: kernel_src + host_tiling_src + host_operator_src + python_bind_src

Usage:
    python 3_basic_evaluation.py          # fake mode only
    python 3_basic_evaluation.py --real   # real NPU evaluation
"""

import argparse

from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution
from _config import (
    KERNEL_SRC,
    BLOCK_DIM,
    HOST_TILING_SRC,
    HOST_OPERATOR_SRC,
    PYTHON_BIND_SRC,
    get_task_data,
    ensure_output_dir,
)


def test_fake_mode():
    """Test in fake mode (writes files but skips compile/NPU)."""
    print("\n" + "-" * 40)
    print("Fake Mode Test (Write Files Only)")
    print("-" * 40)

    output_dir = ensure_output_dir("3_fake")

    task = CANNInitTask(data=get_task_data(), fake_mode=True)

    print(f"Task Type: {task.get_task_type()}")
    print(f"Op Name: {task.op_name}")

    # Test Default mode
    print("\n[Default Mode]")
    config = CANNSolutionConfig(
        project_path=str(output_dir / "default_mode"),
        block_dim=BLOCK_DIM,
    )
    solution = Solution(sol_string=KERNEL_SRC, other_info=config.to_dict())
    result = task.evaluate_solution(solution)
    print(f"  Valid: {result.valid}")
    print(f"  Stage: {result.additional_info.get('stage')}")
    print(f"  Project: {result.additional_info.get('project_path')}")
    if result.valid:
        components = result.additional_info.get('generated_components', [])
        print(f"  Components: {components}")
        files = result.additional_info.get('files_written', [])
        print(f"  Files written: {len(files)}")
    else:
        print(f"  Error: {result.additional_info.get('error')}")

    # Test Full LLM mode
    print("\n[Full LLM Mode]")
    config = CANNSolutionConfig(
        project_path=str(output_dir / "full_llm_mode"),
        host_tiling_src=HOST_TILING_SRC,
        host_operator_src=HOST_OPERATOR_SRC,
        python_bind_src=PYTHON_BIND_SRC,
    )
    solution = Solution(sol_string=KERNEL_SRC, other_info=config.to_dict())
    result = task.evaluate_solution(solution)
    print(f"  Valid: {result.valid}")
    print(f"  Stage: {result.additional_info.get('stage')}")
    print(f"  Project: {result.additional_info.get('project_path')}")
    if result.valid:
        components = result.additional_info.get('generated_components', [])
        print(f"  Components: {components}")
        files = result.additional_info.get('files_written', [])
        print(f"  Files written: {len(files)}")
    else:
        print(f"  Error: {result.additional_info.get('error')}")


def test_real_default_mode(task, output_dir):
    """Test Default mode with real NPU."""
    print("\n" + "-" * 40)
    print("Real Mode: Default (kernel_src only)")
    print("-" * 40)

    config = CANNSolutionConfig(
        project_path=str(output_dir / "default_mode"),
        block_dim=BLOCK_DIM,
    )
    solution = Solution(sol_string=KERNEL_SRC, other_info=config.to_dict())

    print("Evaluating (this may take a few minutes)...")
    result = task.evaluate_solution(solution)

    print(f"Valid: {result.valid}")
    print(f"Stage: {result.additional_info.get('stage')}")

    if result.valid:
        runtime = result.additional_info.get('runtime')
        if runtime:
            print(f"Runtime: {runtime:.4f} ms")
    else:
        print(f"Error: {result.additional_info.get('error')}")

    return result


def test_real_full_llm_mode(task, output_dir):
    """Test Full LLM mode with real NPU."""
    print("\n" + "-" * 40)
    print("Real Mode: Full LLM (kernel + tiling + host + pybind)")
    print("-" * 40)

    config = CANNSolutionConfig(
        project_path=str(output_dir / "full_llm_mode"),
        host_tiling_src=HOST_TILING_SRC,
        host_operator_src=HOST_OPERATOR_SRC,
        python_bind_src=PYTHON_BIND_SRC,
    )
    solution = Solution(sol_string=KERNEL_SRC, other_info=config.to_dict())

    print("Evaluating (this may take a few minutes)...")
    result = task.evaluate_solution(solution)

    print(f"Valid: {result.valid}")
    print(f"Stage: {result.additional_info.get('stage')}")

    if result.valid:
        runtime = result.additional_info.get('runtime')
        if runtime:
            print(f"Runtime: {runtime:.4f} ms")
    else:
        print(f"Error: {result.additional_info.get('error')}")

    return result


def test_real_mode(npu_type: str):
    """Test with real NPU evaluation."""
    print("\n" + "=" * 50)
    print("Real NPU Evaluation")
    print("=" * 50)

    output_dir = ensure_output_dir("3_real")

    task = CANNInitTask(
        data=get_task_data(npu_type=npu_type),
        fake_mode=False,
    )

    # Test both modes
    default_result = test_real_default_mode(task, output_dir)
    full_llm_result = test_real_full_llm_mode(task, output_dir)

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Default Mode:  {'PASS' if default_result.valid else 'FAIL'}")
    print(f"Full LLM Mode: {'PASS' if full_llm_result.valid else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Run real NPU evaluation")
    parser.add_argument("--npu", default="Ascend910B", help="NPU type")
    args = parser.parse_args()

    print("=" * 50)
    print("Basic Evaluation Test")
    print("=" * 50)

    test_fake_mode()

    if args.real:
        test_real_mode(args.npu)


if __name__ == "__main__":
    main()
