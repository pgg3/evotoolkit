# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import argparse

from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution
from _config import (
    KERNEL_SRC,
    TILING_FIELDS,
    TILING_FUNC_BODY,
    INFER_SHAPE_BODY,
    INFER_DTYPE_BODY,
    get_task_data,
    ensure_output_dir,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", default="Ascend910B", help="NPU type")
    args = parser.parse_args()

    print("=" * 50)
    print("Basic Evaluation Test")
    print("=" * 50)

    output_dir = ensure_output_dir("3_evaluation")

    task = CANNInitTask(
        data=get_task_data(npu_type=args.npu),
        fake_mode=False,
    )

    print(f"Task Type: {task.get_task_type()}")
    print(f"Op Name: {task.op_name}")

    config = CANNSolutionConfig(
        project_path=str(output_dir),
        tiling_fields=TILING_FIELDS,
        tiling_func_body=TILING_FUNC_BODY,
        infer_shape_body=INFER_SHAPE_BODY,
        infer_dtype_body=INFER_DTYPE_BODY,
    )
    solution = Solution(sol_string=KERNEL_SRC, other_info=config.to_dict())

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


if __name__ == "__main__":
    main()
