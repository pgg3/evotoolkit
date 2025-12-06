# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Parallel Compilation

Demonstrates parallel compilation of multiple kernel variants,
then sequential testing on NPU.

Usage:
    python 4_parallel_compile.py          # fake mode
    python 4_parallel_compile.py --real   # real NPU
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution
from _config import KERNEL_SRC, get_task_data, ensure_output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--npu", default="Ascend910B")
    args = parser.parse_args()

    fake_mode = not args.real

    print("=" * 50)
    print("Parallel Compilation Test")
    print(f"Mode: {'fake' if fake_mode else 'real'}")
    print("=" * 50)

    output_dir = ensure_output_dir("parallel")

    task = CANNInitTask(
        data=get_task_data(npu_type=args.npu),
        fake_mode=fake_mode,
    )

    # Create solutions with different block_dim
    solutions = []
    for i, block_dim in enumerate([4, 8, 16, 32]):
        config = CANNSolutionConfig(
            project_path=str(output_dir / f"sol_{i:03d}"),
            block_dim=block_dim,
            compile_only=True,
            save_compile_to=str(output_dir / f"sol_{i:03d}"),
        )
        solutions.append((i, block_dim, Solution(KERNEL_SRC, config.to_dict())))

    # Parallel compilation
    print(f"\nCompiling {len(solutions)} solutions in parallel...")
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(task.evaluate_solution, sol): (idx, bd)
            for idx, bd, sol in solutions
        }
        for future in as_completed(futures):
            idx, bd = futures[future]
            result = future.result()
            results.append((idx, bd, result))
            print(f"  sol_{idx} (block_dim={bd}): valid={result.valid}")

    print(f"\nAll {len(results)} compilations done!")

    # Sequential testing (real mode only)
    if not fake_mode:
        print("\nSequential testing...")
        for idx, bd, _ in sorted(results):
            test_result = task.test_compiled(
                load_from=str(output_dir / f"sol_{idx:03d}")
            )
            if test_result.valid:
                rt = test_result.additional_info.get("runtime")
                print(f"  sol_{idx} (block_dim={bd}): {rt:.4f} ms")
            else:
                print(f"  sol_{idx}: failed")


if __name__ == "__main__":
    main()
