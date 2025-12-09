#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Joint Branch Implementation 测试

单独测试代码实现阶段 (Phase 3):
1. tiling.h (tiling data structure)
2. op_host.cpp (tiling calculation + InferShape)
3. op_kernel.cpp (kernel implementation)

输入: joint_plan, knowledge_context (来自 4_joint_planning.py)
输出: tiling_src, operator_src, kernel_src

用法:
    python 5_joint_impl.py [easy|medium|hard]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_llm, get_test_config, load_python_ref, ensure_output_dir,
    get_phase0_context, get_joint_plan_context, get_knowledge_context
)

from evotoolkit.task.cann_init import CANNIniterInterface, CANNInitTask
from evotoolkit.evo_method.cann_initer import CANNIniterConfig
from evotoolkit.evo_method.cann_initer.phases import JointBranch
from evotoolkit.evo_method.cann_initer.run_state_dict import CANNIniterRunStateDict


def main(test_case: str = "hard"):
    config_info = get_test_config(test_case)
    python_ref = load_python_ref(test_case)

    print("=" * 70)
    print(f"Joint Branch Implementation Test - {config_info['name']}")
    print("=" * 70)

    # Get contexts
    try:
        phase0_ctx = get_phase0_context(test_case)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return

    try:
        joint_plan = get_joint_plan_context(test_case)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return

    knowledge = get_knowledge_context(test_case)
    op_name = phase0_ctx["op_name"]
    tiling_strategy = joint_plan.get("tiling_strategy", "default")

    # Initialize
    print("\n[1] Initializing components...")
    print(f"    Tiling strategy: {tiling_strategy}")
    print(f"    Tiling fields: {joint_plan.get('tiling_fields', [])}")

    task = CANNInitTask(data={
        "op_name": op_name,
        "npu_type": "Ascend910B2",
        "python_reference": python_ref,
    })
    llm = get_llm()
    interface = CANNIniterInterface()

    config = CANNIniterConfig(
        task=task,
        interface=interface,
        output_path=str(ensure_output_dir(f"impl_{test_case}")),
        running_llm=llm,
        knowledge_base=None,
        verbose=True,
    )

    # Restore state from Phase 0 + Planning
    run_state_dict = CANNIniterRunStateDict()
    run_state_dict.op_name = phase0_ctx["op_name"]
    run_state_dict.signature = phase0_ctx["signature"]
    run_state_dict.compute_pattern = phase0_ctx["compute_pattern"]
    run_state_dict.strategies = phase0_ctx["strategies"]
    run_state_dict.joint_plan = joint_plan
    run_state_dict.knowledge_context = knowledge

    # Run Implementation (Phase 3 only)
    print("\n[2] Running Joint Branch Implementation...")
    print("-" * 70)
    joint_branch = JointBranch(config, run_state_dict)
    joint_branch.run_implementation(python_ref)

    # Print results
    print("\n" + "=" * 70)
    print("Implementation Results")
    print("=" * 70)

    print(f"  - tiling.h: {'Yes' if run_state_dict.tiling_src else 'No (default tiling)'}")
    print(f"  - op_host.cpp: {'Yes' if run_state_dict.operator_src else 'No (default tiling)'}")
    print(f"  - op_kernel.cpp: {'Yes' if run_state_dict.kernel_src else 'No'}")

    if run_state_dict.tiling_src:
        print(f"\n--- tiling.h ({len(run_state_dict.tiling_src)} chars) ---")
        print(run_state_dict.tiling_src[:500] + "..." if len(run_state_dict.tiling_src) > 500 else run_state_dict.tiling_src)

    if run_state_dict.operator_src:
        print(f"\n--- op_host.cpp ({len(run_state_dict.operator_src)} chars) ---")
        print(run_state_dict.operator_src[:1000] + "..." if len(run_state_dict.operator_src) > 1000 else run_state_dict.operator_src)

    if run_state_dict.kernel_src:
        print(f"\n--- op_kernel.cpp ({len(run_state_dict.kernel_src)} chars) ---")
        print(run_state_dict.kernel_src[:1500] + "..." if len(run_state_dict.kernel_src) > 1500 else run_state_dict.kernel_src)

    # Save outputs
    output_dir = ensure_output_dir(f"impl_{test_case}")
    if run_state_dict.tiling_src:
        (output_dir / "tiling.h").write_text(run_state_dict.tiling_src)
    if run_state_dict.operator_src:
        (output_dir / "op_host.cpp").write_text(run_state_dict.operator_src)
    if run_state_dict.kernel_src:
        (output_dir / "op_kernel.cpp").write_text(run_state_dict.kernel_src)

    print(f"\n[Saved to {output_dir}/]")


if __name__ == "__main__":
    test_case = sys.argv[1] if len(sys.argv) > 1 else "hard"
    main(test_case)
