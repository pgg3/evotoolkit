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
    修改了文件匹配逻辑以符合您的自定义命名。

    Mapping:
    - kernel_src.cpp        -> kernel_src
    - host_tiling_src.h     -> tiling_src
    - host_operator_src.cpp -> operator_src
    - python_bind_src.cpp   -> pybind_src

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

    # 定义代码键值与磁盘文件名的映射关系
    # "kernel_src": "kernel_src.cpp"
    file_mapping = {
        "kernel_src": "kernel_src.cpp",          # 对应 op_kernel.cpp 的逻辑
        "tiling_src": "host_tiling_src.h",       # 对应 tiling.h 的逻辑
        "operator_src": "host_operator_src.cpp", # 对应 op_host.cpp 的逻辑
        "pybind_src": "python_bind_src.cpp"      # 对应 pybind_src.cpp 的逻辑
    }

    for key, filename in file_mapping.items():
        file_path = impl_dir / filename
        if file_path.exists():
            code[key] = file_path.read_text()
        else:
            # 可选：如果找不到新文件名，尝试回退查找旧文件名（增加鲁棒性）
            fallback_names = {
                "kernel_src": "op_kernel.cpp",
                "tiling_src": "tiling.h",
                "operator_src": "op_host.cpp",
                "pybind_src": "pybind_src.cpp"
            }
            fallback_path = impl_dir / fallback_names[key]
            if fallback_path.exists():
                code[key] = fallback_path.read_text()
                print(f"[WARN] Used fallback file for {key}: {fallback_names[key]}")

    return code


def evaluate_full_mode(task, code: dict, output_dir: Path):
    """
    Full LLM mode: kernel + tiling + host + pybind
    必须提供所有组件，不使用默认模板。
    """
    print("\n" + "-" * 50)
    print("Full Mode Evaluation (kernel + tiling + host + pybind)")
    print("-" * 50)

    # Check all required components
    missing = []
    if not code["kernel_src"]:
        missing.append("kernel_src (op_kernel.cpp)")
    if not code["tiling_src"]:
        missing.append("tiling_src (tiling.h)")
    if not code["operator_src"]:
        missing.append("operator_src (op_host.cpp)")
    if not code["pybind_src"]:
        missing.append("pybind_src (pybind_src.cpp)")

    if missing:
        print(f"[ERROR] Missing required files:")
        for m in missing:
            print(f"  - {m}")
        return None

    print(f"  kernel_src: {len(code['kernel_src'])} chars")
    print(f"  tiling_src: {len(code['tiling_src'])} chars")
    print(f"  operator_src: {len(code['operator_src'])} chars")
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
    print(f"    Source: {impl_dir}/")

    for key, val in code.items():
        if val:
            print(f"    - {key}: {len(val)} chars")
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

    # Evaluate (Full mode only)
    print("\n[3] Running evaluation...")
    result = evaluate_full_mode(task, code, output_dir)

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
    parser.add_argument("test_case", nargs="?", default="easy",
                       choices=["easy", "medium", "hard"],
                       help="Test case to evaluate")
    parser.add_argument("--npu", default="Ascend910B2",
                       help="NPU type (default: Ascend910B2)")
    args = parser.parse_args()

    main(args.test_case, args.npu)
