#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Agent 生成代码评估测试

使用 Evaluator 验证 Agent 生成的代码:
- 读取 impl_{test_case}/ 目录下的生成文件
- 通过 CANNInitTask 编译和运行测试
- 输出正确性结果

前置条件:
- 运行 3_pybind.py 生成 pybind_src.cpp
- 运行 5_joint_impl.py 生成 tiling.h, op_host.cpp, op_kernel.cpp

用法:
    python 7_evaluate.py [easy|medium|hard]
    python 7_evaluate.py hard --npu Ascend910B2
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_test_config, load_python_ref, ensure_output_dir
)

from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution


def load_generated_code(test_case: str) -> dict:
    """
    从 impl_{test_case}/ 目录加载生成的代码文件。

    Returns:
        dict with keys: kernel_src, tiling_src, operator_src, pybind_src
    """
    impl_dir = ensure_output_dir(f"impl_{test_case}")

    code = {
        "kernel_src": None,
        "tiling_src": None,
        "operator_src": None,
        "pybind_src": None,
    }

    # Load files if they exist
    kernel_file = impl_dir / "op_kernel.cpp"
    if kernel_file.exists():
        code["kernel_src"] = kernel_file.read_text()

    tiling_file = impl_dir / "tiling.h"
    if tiling_file.exists():
        code["tiling_src"] = tiling_file.read_text()

    operator_file = impl_dir / "op_host.cpp"
    if operator_file.exists():
        code["operator_src"] = operator_file.read_text()

    pybind_file = impl_dir / "pybind_src.cpp"
    if pybind_file.exists():
        code["pybind_src"] = pybind_file.read_text()

    return code


def evaluate_default_mode(task, code: dict, output_dir: Path):
    """
    Default mode: 仅使用 kernel_src (element-wise 算子)
    """
    print("\n" + "-" * 50)
    print("Mode 1: Default (kernel_src only)")
    print("-" * 50)

    if not code["kernel_src"]:
        print("[SKIP] kernel_src not found")
        return None

    config = CANNSolutionConfig(
        project_path=str(output_dir / "default_mode"),
        block_dim=8,
    )
    solution = Solution(sol_string=code["kernel_src"], other_info=config.to_dict())

    print("Evaluating...")
    result = task.evaluate_solution(solution)

    print(f"  Valid: {result.valid}")
    print(f"  Stage: {result.additional_info.get('stage')}")

    if result.valid:
        runtime = result.additional_info.get('runtime')
        if runtime:
            print(f"  Runtime: {runtime:.4f} ms")
    else:
        error = result.additional_info.get('error', '')
        print(f"  Error: {error[:200]}..." if len(error) > 200 else f"  Error: {error}")

    return result


def evaluate_full_mode(task, code: dict, output_dir: Path):
    """
    Full LLM mode: kernel + tiling + host + pybind
    """
    print("\n" + "-" * 50)
    print("Mode 2: Full LLM (kernel + tiling + host + pybind)")
    print("-" * 50)

    if not code["kernel_src"]:
        print("[SKIP] kernel_src not found")
        return None

    # Check what we have
    has_tiling = code["tiling_src"] is not None
    has_operator = code["operator_src"] is not None
    has_pybind = code["pybind_src"] is not None

    print(f"  kernel_src: Yes")
    print(f"  tiling_src: {'Yes' if has_tiling else 'No (default)'}")
    print(f"  operator_src: {'Yes' if has_operator else 'No (default)'}")
    print(f"  pybind_src: {'Yes' if has_pybind else 'No (default)'}")

    config = CANNSolutionConfig(
        project_path=str(output_dir / "full_mode"),
        host_tiling_src=code["tiling_src"],
        host_operator_src=code["operator_src"],
        python_bind_src=code["pybind_src"],
    )
    solution = Solution(sol_string=code["kernel_src"], other_info=config.to_dict())

    print("\nEvaluating...")
    result = task.evaluate_solution(solution)

    print(f"  Valid: {result.valid}")
    print(f"  Stage: {result.additional_info.get('stage')}")

    if result.valid:
        runtime = result.additional_info.get('runtime')
        if runtime:
            print(f"  Runtime: {runtime:.4f} ms")
    else:
        error = result.additional_info.get('error', '')
        print(f"  Error: {error[:200]}..." if len(error) > 200 else f"  Error: {error}")

    return result


def main(test_case: str = "hard", npu_type: str = "Ascend910B2"):
    config_info = get_test_config(test_case)
    python_ref = load_python_ref(test_case)
    op_name = config_info["op_name"]

    print("=" * 60)
    print(f"Agent Code Evaluation - {config_info['name']}")
    print("=" * 60)

    # Load generated code
    print("\n[1] Loading generated code...")
    code = load_generated_code(test_case)

    impl_dir = ensure_output_dir(f"impl_{test_case}")
    print(f"    Source: {impl_dir}/")

    for key, val in code.items():
        if val:
            print(f"    - {key}: {len(val)} chars")
        else:
            print(f"    - {key}: (not found)")

    if not code["kernel_src"]:
        print("\n[ERROR] kernel_src not found. Please run 5_joint_impl.py first.")
        return

    # Create task
    print("\n[2] Creating evaluation task...")
    output_dir = ensure_output_dir(f"eval_{test_case}")
    task = CANNInitTask(
        data={
            "op_name": op_name,
            "npu_type": npu_type,
            "python_reference": python_ref,
        },
        fake_mode=False,
    )
    print(f"    Op: {op_name}")
    print(f"    NPU: {npu_type}")

    # Evaluate
    print("\n[3] Running evaluation...")
    default_result = evaluate_default_mode(task, code, output_dir)
    full_result = evaluate_full_mode(task, code, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if default_result:
        print(f"  Default Mode:  {'PASS' if default_result.valid else 'FAIL'}")
    else:
        print(f"  Default Mode:  SKIP")

    if full_result:
        print(f"  Full Mode:     {'PASS' if full_result.valid else 'FAIL'}")
    else:
        print(f"  Full Mode:     SKIP")

    # Return success status
    success = (full_result and full_result.valid) or (default_result and default_result.valid)
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent-generated CANN code")
    parser.add_argument("test_case", nargs="?", default="hard",
                       choices=["easy", "medium", "hard"],
                       help="Test case to evaluate")
    parser.add_argument("--npu", default="Ascend910B2",
                       help="NPU type (default: Ascend910B2)")
    args = parser.parse_args()

    main(args.test_case, args.npu)
