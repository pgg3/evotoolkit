# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Basic Evaluation

Demonstrates CANNInitTask evaluation in fake and real modes.

Usage:
    python 3_basic_evaluation.py          # fake mode
    python 3_basic_evaluation.py --real   # real NPU
"""

import argparse

from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution
from _config import KERNEL_SRC, get_task_data, ensure_output_dir


def test_fake_mode():
    """Test in fake mode (no NPU required)."""
    print("\n" + "-" * 40)
    print("Fake Mode Test")
    print("-" * 40)

    task = CANNInitTask(data=get_task_data(), fake_mode=True)

    print(f"Task Type: {task.get_task_type()}")
    print(f"Op Name: {task.op_name}")

    result = task.evaluate_code(KERNEL_SRC)
    print(f"Valid: {result.valid}")
    print(f"Score: {result.score}")


def test_real_mode(npu_type: str):
    """Test with real NPU evaluation."""
    print("\n" + "-" * 40)
    print("Real Mode Test")
    print("-" * 40)

    output_dir = ensure_output_dir("basic")

    task = CANNInitTask(
        data=get_task_data(npu_type=npu_type),
        fake_mode=False,
    )

    solution = Solution(
        sol_string=KERNEL_SRC,
        other_info={"project_path": str(output_dir)},
    )

    print("Evaluating (this may take a few minutes)...")
    result = task.evaluate_solution(solution)

    print(f"Valid: {result.valid}")
    print(f"Stage: {result.additional_info.get('stage')}")

    if result.valid:
        print(f"Runtime: {result.additional_info.get('runtime'):.4f} ms")
    else:
        print(f"Error: {result.additional_info.get('error')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--npu", default="Ascend910B")
    args = parser.parse_args()

    print("=" * 50)
    print("Basic Evaluation Test")
    print("=" * 50)

    test_fake_mode()

    if args.real:
        test_real_mode(args.npu)


if __name__ == "__main__":
    main()
