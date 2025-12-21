#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Agent 生成代码评估测试

使用 Evaluator 验证 Agent 生成的代码:
- 读取 impl_{test_case}/ 目录下的 kernel/tiling/operator 文件
- 读取 pybind_{test_case}/ 目录下的 pybind_src.cpp
- 通过 CANNInitTask 编译和运行测试
- 输出正确性结果

支持两种模式:
1. Default tiling: 只需 kernel_src + pybind_src (tiling/operator 使用默认模板)
2. Custom tiling: 需要全部 4 个文件

前置条件:
- 运行 3_pybind.py 生成 pybind_src.cpp
- 运行 5_joint_impl.py 生成 op_kernel.cpp (+ 可选的 tiling.h, op_host.cpp)

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
    加载生成的代码文件。

    Sources:
    - kernel_src, tiling_src, operator_src: impl_{test_case}/
    - pybind_src: pybind_{test_case}/

    Returns:
        dict with keys: kernel_src, tiling_src, operator_src, pybind_src
    """
    impl_dir = ensure_output_dir(f"impl_{test_case}")
    pybind_dir = ensure_output_dir(f"pybind_{test_case}")

    code = {
        "kernel_src": None,
        "tiling_src": None,
        "operator_src": None,
        "pybind_src": None,
    }

    # Load kernel/tiling/operator from impl_dir
    kernel_file = impl_dir / "op_kernel.cpp"
    if kernel_file.exists():
        code["kernel_src"] = kernel_file.read_text()

    tiling_file = impl_dir / "tiling.h"
    if tiling_file.exists():
        code["tiling_src"] = tiling_file.read_text()

    operator_file = impl_dir / "op_host.cpp"
    if operator_file.exists():
        code["operator_src"] = operator_file.read_text()

    # Load pybind_src from pybind_dir
    pybind_file = pybind_dir / "pybind_src.cpp"
    if pybind_file.exists():
        code["pybind_src"] = pybind_file.read_text()

    return code


def evaluate_generated_code(task, code: dict, output_dir: Path):
    """
    评估生成的代码。

    支持两种模式：
    1. Default tiling: kernel + pybind (tiling_src 和 operator_src 为 None)
    2. Custom tiling: kernel + tiling + host + pybind (全部提供)
    """
    # Determine mode
    is_default_tiling = code["tiling_src"] is None and code["operator_src"] is None
    mode_name = "Default Tiling" if is_default_tiling else "Custom Tiling"

    print("\n" + "-" * 50)
    print(f"Evaluation Mode: {mode_name}")
    print("-" * 50)

    # Check required components
    missing = []
    if not code["kernel_src"]:
        missing.append("kernel_src (op_kernel.cpp)")
    if not code["pybind_src"]:
        missing.append("pybind_src (pybind_src.cpp)")

    # For custom tiling, also require tiling and operator
    if not is_default_tiling:
        if not code["tiling_src"]:
            missing.append("tiling_src (tiling.h)")
        if not code["operator_src"]:
            missing.append("operator_src (op_host.cpp)")

    if missing:
        print(f"[ERROR] Missing required files:")
        for m in missing:
            print(f"  - {m}")
        return None

    print(f"  kernel_src: {len(code['kernel_src'])} chars")
    print(f"  tiling_src: {len(code['tiling_src']) if code['tiling_src'] else 'None (default)'}")
    print(f"  operator_src: {len(code['operator_src']) if code['operator_src'] else 'None (default)'}")
    print(f"  pybind_src: {len(code['pybind_src'])} chars")

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
        # Print full error for debugging
        print(f"  Error:\n{error}")

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
    pybind_dir = ensure_output_dir(f"pybind_{test_case}")
    print(f"    Sources:")
    print(f"      - kernel/tiling/operator: {impl_dir}/")
    print(f"      - pybind: {pybind_dir}/")

    for key, val in code.items():
        if val:
            print(f"    - {key}: {len(val)} chars")
        elif key in ("tiling_src", "operator_src"):
            print(f"    - {key}: (None - using default)")
        else:
            print(f"    - {key}: (not found)")

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
    result = evaluate_generated_code(task, code, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if result:
        status = 'PASS' if result.valid else 'FAIL'
        print(f"  Result: {status}")
    else:
        print(f"  Result: SKIP (missing files)")

    return result and result.valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent-generated CANN code")
    parser.add_argument("test_case", nargs="?", default="hard",
                       choices=["easy", "medium", "hard"],
                       help="Test case to evaluate")
    parser.add_argument("--npu", default="Ascend910B2",
                       help="NPU type (default: Ascend910B2)")
    args = parser.parse_args()

    main(args.test_case, args.npu)
