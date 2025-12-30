# Copyright (c) 2025 Sun Yansong
# Licensed under the MIT License

"""
测试: Iterative Debug Loop (独立调试循环)

目的:
    读取 impl_{test_case} 目录下的现有代码(可能包含错误)，
    直接启动 DebugLoop 进行 "编译-报错-修复" 的闭环测试。

用途:
    1. 测试 LLM 的代码修复能力 (Repair Capability)。
    2. 验证 debug_loop.py 的逻辑修改是否有效 (例如 Context 是否传全)。
    3. 在不重新生成代码的情况下，抢救一个写坏了的算子。

用法:
    python 8_iter_debug.py [easy|medium|hard]
    python 8_iter_debug.py easy --npu Ascend910B2
"""

import sys
import argparse
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_llm, get_test_config, load_python_ref, ensure_output_dir, get_knowledge_base
)

from evotoolkit.task.cann_init import CANNInitTask, CANNIniterInterface
from evotoolkit.evo_method.cann_initer import CANNIniterConfig
from evotoolkit.evo_method.cann_initer.run_state_dict import CANNIniterRunStateDict
# 引入 DebugLoop
from evotoolkit.evo_method.cann_initer.phases.debug_loop import DebugLoop


def load_broken_code(test_case: str) -> dict:
    """
    从 impl_{test_case}/ 目录加载代码。
    包含文件名映射逻辑，适配 generated code 的命名习惯。
    """
    impl_dir = ensure_output_dir(f"impl_{test_case}")
    print(f"[Info] Loading code from: {impl_dir}")

    code = {
        "kernel_src": None,
        "tiling_src": None,
        "operator_src": None,
        "pybind_src": None,
    }

    # 映射关系: 内部Key -> 磁盘文件名
    # 优先读取 impl_easy 生成的文件名
    file_mapping = {
        "kernel_src": ["kernel_src.cpp", "op_kernel.cpp"],
        "tiling_src": ["host_tiling_src.h", "tiling.h"],
        "operator_src": ["host_operator_src.cpp", "op_host.cpp"],
        "pybind_src": ["python_bind_src.cpp", "pybind_src.cpp"]
    }

    for key, filenames in file_mapping.items():
        found = False
        for fname in filenames:
            file_path = impl_dir / fname
            if file_path.exists():
                code[key] = file_path.read_text()
                print(f"  - Loaded {key} from {fname} ({len(code[key])} chars)")
                found = True
                break
        if not found:
            print(f"  [WARN] Missing file for {key} (checked: {filenames})")

    return code


def main():
    max_debug_iterations = 8
    parser = argparse.ArgumentParser(description="Run Debug Loop independently")
    parser.add_argument("test_case", nargs="?", default="easy",
                        choices=["easy", "medium", "hard"],
                        help="Test case to debug")
    parser.add_argument("--npu", default="Ascend910B2",
                        help="NPU type (default: Ascend910B2)")
    args = parser.parse_args()

    # 1. 准备配置和数据
    config_info = get_test_config(args.test_case)
    python_ref = load_python_ref(args.test_case)
    op_name = config_info["op_name"]

    print("=" * 60)
    print(f"Iterative Debug Test - {op_name}")
    print("=" * 60)

    # 2. 加载代码 (可以是坏的代码)
    code_dict = load_broken_code(args.test_case)

    # 检查核心代码是否存在
    if not code_dict["kernel_src"]:
        print("\n[Error] Cannot start debug: kernel_src is missing!")
        return

    # 3. 初始化组件
    print("\n[1] Initializing Agent Components...")

    # 任务环境 (用于编译和运行)
    task = CANNInitTask(
        data={
            "op_name": op_name,
            "npu_type": args.npu,
            "python_reference": python_ref,
        },
        fake_mode=False
    )

    # LLM 修复代码
    llm = get_llm()
    interface = CANNIniterInterface()

    # 配置对象
    kb = get_knowledge_base()
    config = CANNIniterConfig(
        task=task,
        interface=interface,
        output_path=str(ensure_output_dir(f"debug_fixed_{args.test_case}")),  # 结果存到新目录
        running_llm=llm,
        # knowledge_base = kb,   # 查库
        # Debug阶段不查库，只看报错
        knowledge_base = None,
        verbose=True,
        max_debug_iterations = max_debug_iterations
    )

    # 4. 初始化状态字典 (填入加载的代码)
    run_state_dict = CANNIniterRunStateDict()
    run_state_dict.op_name = op_name
    run_state_dict.kernel_src = code_dict.get("kernel_src")
    run_state_dict.tiling_src = code_dict.get("tiling_src")
    run_state_dict.operator_src = code_dict.get("operator_src")
    run_state_dict.pybind_src = code_dict.get("pybind_src")

    # 5. 启动 Debug Loop
    print("\n[2] Starting Debug Loop...")
    print("-" * 60)

    debugger = DebugLoop(config, run_state_dict)

    # 运行!
    result = debugger.run(python_ref)

    # 6. 结果摘要
    print("\n" + "=" * 60)
    print("Debug Result Summary")
    print("=" * 60)
    print(f"Success: {result['success']}")

    output_dir = ensure_output_dir(f"debug_fixed_{args.test_case}")
    print(f"Fixed code saved to: {output_dir}")

    # 保存最终代码
    final_code = result["code"]
    if final_code.get("kernel_src"):
        (output_dir / "op_kernel.cpp").write_text(final_code["kernel_src"])
    if final_code.get("host_tiling_src"):
        (output_dir / "tiling.h").write_text(final_code["host_tiling_src"])
    if final_code.get("host_operator_src"):
        (output_dir / "op_host.cpp").write_text(final_code["host_operator_src"])
    if final_code.get("python_bind_src"):
        (output_dir / "pybind_src.cpp").write_text(final_code["python_bind_src"])


if __name__ == "__main__":
    main()