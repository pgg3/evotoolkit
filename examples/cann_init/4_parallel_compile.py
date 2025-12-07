# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Parallel Compilation

Demonstrates parallel compilation of multiple kernel variants,
then sequential testing on NPU.

Usage:
    python 4_parallel_compile.py
    python 4_parallel_compile.py --npu Ascend910B
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution
from _config import KERNEL_SRC, get_task_data, ensure_output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", default="Ascend910B")
    args = parser.parse_args()

    print("=" * 50)
    print("Parallel Compilation Test")
    print("=" * 50)

    output_dir = ensure_output_dir("4_parallel")

    task = CANNInitTask(
        data=get_task_data(npu_type=args.npu),
        fake_mode=False,
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
            status = "compiled" if result.valid else "failed"
            print(f"  sol_{idx} (block_dim={bd}): {status}")

    compiled_count = sum(1 for _, _, r in results if r.valid)
    print(f"\nCompilation: {compiled_count}/{len(results)} succeeded")

    # Sequential testing
    print("\nSequential testing...")
    for idx, bd, compile_result in sorted(results):
        if not compile_result.valid:
            print(f"  sol_{idx} (block_dim={bd}): skipped (compile failed)")
            continue

        test_result = task.test_compiled(
            load_from=str(output_dir / f"sol_{idx:03d}")
        )
        if test_result.valid:
            rt = test_result.additional_info.get("runtime")
            print(f"  sol_{idx} (block_dim={bd}): {rt:.4f} ms")
        else:
            err = test_result.additional_info.get("error", "unknown")
            print(f"  sol_{idx} (block_dim={bd}): failed - {err[:50]}")


if __name__ == "__main__":
    main()
