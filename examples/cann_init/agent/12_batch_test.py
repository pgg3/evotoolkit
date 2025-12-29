#!/usr/bin/env python3
# Copyright (c) 2025 Yansong Sun
# Licensed under the MIT License

"""
CANNIniter Batch Test for Element-wise Operators
批量测试符合 "Element-wise operation, uses default tiling" 标准的算子。

此脚本不依赖 _config.py 中的 TEST_CASES 定义，而是直接加载文件进行测试，
避免了与现有的 easy/medium/hard 配置冲突。

测试目标:
1. tanh.py
2. sigmoid.py
3. gelu.py
4. min_gpt_new_gelu.py  更加复杂的复合数学公式
下一轮：
1. matrix_scalar_multiplication.py (Scalar Broadcast)
2. hardsigmoid.py (Element-wise)
3. softsign.py (Element-wise)
4. swish.py (Element-wise, Composition)
5. softplus.py (Element-wise)
6. softmax.py (Reduction - 压力测试 Default Tiling 的局限性)
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_llm,
    get_knowledge_base,
    ensure_output_dir,
    TEST_CASES_DIR
)

from evotoolkit.task.cann_init import CANNIniterInterface, CANNInitTask
from evotoolkit.evo_method.cann_initer import CANNIniter, CANNIniterConfig


# =============================================================================
# 自定义批处理配置
# =============================================================================

@dataclass
class TestCaseDef:
    filename: str
    op_name: str
    description: str
'''
    TestCaseDef(
        filename="tanh.py",
        op_name="tanh",
        description="Standard Tanh activation, uses default tiling"
    ),
    TestCaseDef(
        filename="sigmoid.py",
        op_name="sigmoid",
        description="Standard Sigmoid activation, uses default tiling"
    ),
    TestCaseDef(
        filename="gelu.py",
        op_name="gelu",
        description="Standard GELU activation using PyTorch functional API, uses default tiling"
    ),
    TestCaseDef(
        filename="min_gpt_new_gelu.py",
        op_name="newgelu",
        description="Composite Math: 0.5*x*(1+tanh(...)), tests complex element-wise formula, uses default tiling"
    ),

    
    # 1. Matrix Scalar Multiplication
    # 挑战点：输入包含标量(float)，测试 Agent 是否能将标量正确放入 TilingData 或处理为 1-element Tensor
    TestCaseDef(
        filename="matrix_scalar_multiplication.py",
        op_name="mat_scalar",  # 全小写无下划线
        description="Element-wise: Matrix * Scalar. Tests scalar input handling."
    ),
    # 2. Hard Sigmoid
    # 挑战点：分段线性函数，涉及 Clamp/Min/Max 或 Compare 指令
    TestCaseDef(
        filename="hardsigmoid.py",
        op_name="hard_sigmoid",
        description="Element-wise: Hardsigmoid activation."
    ),
    # 3. Softsign
    # 挑战点：x / (1 + |x|)。涉及 Abs, Div。
    TestCaseDef(
        filename="softsign.py",
        op_name="softsign",
        description="Element-wise: Softsign activation."
    ),

    # 4. Swish
    # 挑战点：x * sigmoid(x)。复合函数，测试 Exp/Div/Mul 组合。
    TestCaseDef(
        filename="swish.py",
        op_name="swish",
        description="Element-wise: Swish activation."
    ),

    # 5. Softplus
    # 挑战点：ln(1 + e^x)。涉及 Exp, Log。测试数值稳定性。
    TestCaseDef(
        filename="softplus.py",
        op_name="softplus",
        description="Element-wise: Softplus activation."
    ),
    '''

# 这里定义你要跑的5个算子
BATCH_CASES = [
    # 3. Softsign
    # 挑战点：x / (1 + |x|)。涉及 Abs, Div。
    TestCaseDef(
        filename="softsign.py",
        op_name="softsign",
        description="Element-wise: Softsign activation."
    ),
    # 2. Hard Sigmoid
    # 挑战点：分段线性函数，涉及 Clamp/Min/Max 或 Compare 指令
    TestCaseDef(
        filename="hardsigmoid.py",
        op_name="hard_sigmoid",
        description="Element-wise: Hardsigmoid activation."
    ),
    # 1. Matrix Scalar Multiplication
    # 挑战点：输入包含标量(float)，测试 Agent 是否能将标量正确放入 TilingData 或处理为 1-element Tensor
    TestCaseDef(
        filename="matrix_scalar_multiplication.py",
        op_name="mat_scalar",  # 全小写无下划线
        description="Element-wise: Matrix * Scalar. Tests scalar input handling."
    ),

]


def run_single_case(case: TestCaseDef, llm, kb, interface):
    """运行单个测试用例"""
    print(f"\n{'#' * 80}")
    print(f"Starting Test Case: {case.op_name}")
    print(f"File: {case.filename}")
    print(f"Description: {case.description}")
    print(f"{'#' * 80}\n")

    # 1. Load Reference Code Manually
    file_path = TEST_CASES_DIR / case.filename
    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return False, {}

    python_ref = file_path.read_text(encoding='utf-8')

    # 2. Initialize Task
    task = CANNInitTask(data={
        "op_name": case.op_name,
        "npu_type": "Ascend910B2",
        "python_reference": python_ref,
    })

    # 3. Create Config
    # 使用独立的输出目录: output/batch_test_{op_name}
    output_subdir = f"batch_test_{case.op_name.lower()}"

    config = CANNIniterConfig(
        task=task,
        interface=interface,
        output_path=str(ensure_output_dir(output_subdir)),
        running_llm=llm,
        knowledge_base=kb,
        verbose=True,
        # Element-wise 通常比较简单，设置较小的轮数即可
        max_debug_iterations=5,
        max_joint_turns=3,
    )

    # 4. Run Initer
    try:
        initer = CANNIniter(config)
        result = initer.run(op_name=case.op_name, python_ref=python_ref)

        # Save generated code
        output_dir = ensure_output_dir(output_subdir)
        for key, code in result['code'].items():
            if code:
                suffix = ".cpp" if key != "host_tiling_src" else ".h"
                filename = f"{key}{suffix}"
                (output_dir / filename).write_text(code, encoding='utf-8')

        return result['success'], result['code']

    except Exception as e:
        print(f"[Exception] Failed to run {case.op_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def main():
    print("Initializing LLM and Knowledge Base...")
    llm = get_llm()
    kb = get_knowledge_base()
    interface = CANNIniterInterface()

    results = []

    print(f"\nPlanned Batch: {[c.op_name for c in BATCH_CASES]}")

    # 批量运行
    for case in BATCH_CASES:
        start_time = time.time()
        success, code_dict = run_single_case(case, llm, kb, interface)
        duration = time.time() - start_time

        results.append({
            "name": case.op_name,
            "success": success,
            "duration": f"{duration:.2f}s",
            "generated_files": list(code_dict.keys()) if code_dict else []
        })

    # 最终报告
    print(f"\n{'=' * 60}")
    print("BATCH TEST REPORT")
    print(f"{'=' * 60}")

    success_count = 0
    for res in results:
        status = "PASS" if res['success'] else "FAIL"
        if res['success']: success_count += 1
        print(f"[{status}] {res['name']:<20} Time: {res['duration']:<10} Files: {len(res['generated_files'])}")

    print(f"\nTotal: {len(results)}, Passed: {success_count}, Failed: {len(results) - success_count}")
    print(f"Outputs are saved in: {ensure_output_dir()}batch_test_*/")


if __name__ == "__main__":
    main()