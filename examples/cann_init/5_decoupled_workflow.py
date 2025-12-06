# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Decoupled Workflow

Demonstrates separated compilation and testing:
1. Compile only
2. Test correctness only
3. Measure performance

Usage:
    python 5_decoupled_workflow.py --real --npu Ascend910B
"""

import argparse

from evotoolkit.task.cann_init import CANNInitTask
from _config import KERNEL_SRC, get_task_data, ensure_output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", required=True,
                        help="Real mode required for this test")
    parser.add_argument("--npu", default="Ascend910B")
    args = parser.parse_args()

    print("=" * 50)
    print("Decoupled Workflow Test")
    print("=" * 50)

    output_dir = ensure_output_dir("decoupled")

    task = CANNInitTask(
        data=get_task_data(npu_type=args.npu),
        fake_mode=False,
    )

    # Step 1: Compile only
    print("\n[Step 1] Compile Only")
    compile_result = task.compile_only(
        kernel_src=KERNEL_SRC,
        project_path=str(output_dir),
        save_to=str(output_dir),
    )
    print(f"Valid: {compile_result.valid}")
    if not compile_result.valid:
        print(f"Error: {compile_result.additional_info.get('error')}")
        return

    # Step 2: Test correctness only
    print("\n[Step 2] Correctness Only")
    correctness_result = task.test_compiled(
        load_from=str(output_dir),
        skip_performance=True,
    )
    print(f"Valid: {correctness_result.valid}")
    if not correctness_result.valid:
        print(f"Error: {correctness_result.additional_info.get('error')}")
        return

    # Step 3: Measure performance
    print("\n[Step 3] Performance")
    perf_result = task.test_compiled(
        load_from=str(output_dir),
        skip_correctness=True,
    )
    print(f"Valid: {perf_result.valid}")
    if perf_result.valid:
        print(f"Runtime: {perf_result.additional_info.get('runtime'):.4f} ms")


if __name__ == "__main__":
    main()
