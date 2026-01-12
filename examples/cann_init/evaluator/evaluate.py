# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANN Operator Evaluation Script

Usage:
    python evaluate.py              # Single evaluation (default)
    python evaluate.py --num 4      # Parallel compilation with 4 solutions
    python evaluate.py --num 4 --delay 3.0  # Custom stagger delay
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution
from _config import (
    KERNEL_IMPL,
    KERNEL_ENTRY_BODY,
    TILING_FIELDS,
    TILING_FUNC_BODY,
    INFER_SHAPE_BODY,
    OUTPUT_ALLOC_CODE,
    get_task_data,
    ensure_output_dir,
)


def build_with_delay(task, sol, delay_seconds):
    time.sleep(delay_seconds)
    return task.evaluate_solution(sol)


def run_single(task, output_dir):
    """Single solution evaluation (no parallelism)."""
    config = CANNSolutionConfig(
        project_path=str(output_dir),
        kernel_impl=KERNEL_IMPL,
        kernel_entry_body=KERNEL_ENTRY_BODY,
        tiling_fields=TILING_FIELDS,
        tiling_func_body=TILING_FUNC_BODY,
        infer_shape_body=INFER_SHAPE_BODY,
        output_alloc_code=OUTPUT_ALLOC_CODE,
    )
    solution = Solution(sol_string="", other_info=config.to_dict())

    print("\nEvaluating (this may take a few minutes)...")
    result = task.evaluate_solution(solution)

    print(f"Valid: {result.valid}")
    print(f"Stage: {result.additional_info.get('stage')}")

    if result.valid:
        runtime = result.additional_info.get('runtime')
        if runtime:
            print(f"Runtime: {runtime:.4f} ms")
    else:
        print(f"Error: {result.additional_info.get('error')}")


def run_parallel(task, output_dir, num_solutions, delay):
    """Parallel compilation with sequential testing."""
    solutions = []
    for i in range(num_solutions):
        config = CANNSolutionConfig(
            project_path=str(output_dir / f"sol_{i:03d}"),
            kernel_impl=KERNEL_IMPL,
            kernel_entry_body=KERNEL_ENTRY_BODY,
            tiling_fields=TILING_FIELDS,
            tiling_func_body=TILING_FUNC_BODY,
            infer_shape_body=INFER_SHAPE_BODY,
            output_alloc_code=OUTPUT_ALLOC_CODE,
            compile_only=True,
            save_compile_to=str(output_dir / f"sol_{i:03d}"),
        )
        solutions.append((i, Solution("", config.to_dict())))

    # Phase 1: Sequential setup
    print(f"\nPhase 1: Sequential project setup ({num_solutions} solutions)...")
    setup_results = []
    for idx, sol in solutions:
        config_dict = sol.other_info.copy()
        config_dict["setup_only"] = True
        setup_sol = Solution(sol.sol_string, config_dict)

        result = task.evaluate_solution(setup_sol)
        status = "ready" if result.valid else "failed"
        print(f"  sol_{idx}: {status}")
        setup_results.append((idx, result))

    ready_solutions = [(idx, r) for idx, r in setup_results if r.valid]
    if not ready_solutions:
        print("\nAll setups failed. Exiting.")
        return

    # Phase 2: Parallel build
    print(f"\nPhase 2: Parallel build with {delay}s stagger ({len(ready_solutions)} solutions)...")
    build_results = []

    with ThreadPoolExecutor(max_workers=num_solutions) as executor:
        futures = {}
        for i, (idx, _) in enumerate(ready_solutions):
            config_dict = {
                "project_path": str(output_dir / f"sol_{idx:03d}"),
                "build_only": True,
                "save_compile_to": str(output_dir / f"sol_{idx:03d}"),
            }
            build_sol = Solution("", config_dict)
            future = executor.submit(build_with_delay, task, build_sol, i * delay)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            status = "built" if result.valid else "failed"
            err_info = ""
            if not result.valid:
                err = result.additional_info.get('error', '')
                err_info = f"\n    Error: {err[:500]}"
            print(f"  sol_{idx}: {status}{err_info}")
            build_results.append((idx, result))

    built_count = sum(1 for _, r in build_results if r.valid)
    print(f"\nBuild: {built_count}/{len(ready_solutions)} succeeded")

    # Phase 3: Sequential testing
    print("\nPhase 3: Sequential testing...")
    for idx, build_result in sorted(build_results):
        if not build_result.valid:
            print(f"  sol_{idx}: skipped (build failed)")
            continue

        config = CANNSolutionConfig(load_from=str(output_dir / f"sol_{idx:03d}"))
        test_sol = Solution("", config.to_dict())
        test_result = task.evaluate_solution(test_sol)

        if test_result.valid:
            rt = test_result.additional_info.get("runtime")
            print(f"  sol_{idx}: {rt:.4f} ms")
        else:
            err = test_result.additional_info.get("error", "unknown")
            print(f"  sol_{idx}: failed - {err[:50]}")


def main():
    parser = argparse.ArgumentParser(description="CANN Operator Evaluation")
    parser.add_argument("--npu", default="Ascend910B", help="NPU type")
    parser.add_argument("--num", type=int, default=1, help="Number of parallel solutions (1=single, >1=parallel)")
    parser.add_argument("--delay", type=float, default=2.0, help="Stagger delay between parallel builds (seconds)")
    args = parser.parse_args()

    print("=" * 50)
    if args.num == 1:
        print("CANN Evaluation (Single)")
    else:
        print(f"CANN Evaluation (Parallel: {args.num} solutions, delay: {args.delay}s)")
    print("=" * 50)

    output_dir = ensure_output_dir("evaluation" if args.num == 1 else "parallel")

    task = CANNInitTask(
        data=get_task_data(npu_type=args.npu),
        fake_mode=False,
    )

    print(f"Task Type: {task.get_task_type()}")
    print(f"Op Name: {task.op_name}")

    if args.num == 1:
        run_single(task, output_dir)
    else:
        run_parallel(task, output_dir, args.num, args.delay)


if __name__ == "__main__":
    main()
