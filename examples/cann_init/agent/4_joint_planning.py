#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Joint Branch Planning 测试

测试 Joint Branch 的规划流程 (不含代码实现):
- Phase 1: Tiling-Kernel 多轮对话
- Phase 2: Knowledge Retrieval

输入: signature, compute_pattern, python_ref (来自 Phase 0)
输出: joint_plan, knowledge_context

用法:
    python 4_joint_planning.py [easy|medium|hard]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_llm, get_knowledge_base, get_test_config, load_python_ref,
    ensure_output_dir, get_phase0_context
)

from evotoolkit.task.cann_init import CANNIniterInterface, CANNInitTask
from evotoolkit.evo_method.cann_initer import CANNIniterConfig
from evotoolkit.evo_method.cann_initer.phases import JointBranch
from evotoolkit.evo_method.cann_initer.run_state_dict import CANNIniterRunStateDict


def main(test_case: str = "hard"):
    config_info = get_test_config(test_case)
    python_ref = load_python_ref(test_case)

    print("=" * 70)
    print(f"Joint Branch Planning Test - {config_info['name']}")
    print("=" * 70)

    # Get Phase 0 context
    try:
        phase0_ctx = get_phase0_context(test_case)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return

    op_name = phase0_ctx["op_name"]

    # Initialize components
    print("\n[1] Initializing components...")
    task = CANNInitTask(data={
        "op_name": op_name,
        "npu_type": "Ascend910B2",
        "python_reference": python_ref,
    })
    llm = get_llm()
    interface = CANNIniterInterface()
    kb = get_knowledge_base()
    print(f"    Knowledge Base: {kb.get_api_count()} APIs, {kb.get_operator_count()} operators")

    config = CANNIniterConfig(
        task=task,
        interface=interface,
        output_path=str(ensure_output_dir(f"joint_{test_case}")),
        running_llm=llm,
        knowledge_base=kb,
        verbose=True,
        max_joint_turns=config_info["max_joint_turns"],
    )

    # Restore Phase 0 state
    run_state_dict = CANNIniterRunStateDict()
    run_state_dict.op_name = phase0_ctx["op_name"]
    run_state_dict.signature = phase0_ctx["signature"]
    run_state_dict.compute_pattern = phase0_ctx["compute_pattern"]
    run_state_dict.strategies = phase0_ctx["strategies"]

    # Run Joint Branch Planning (Phase 1 + 2 only)
    print("\n[2] Running Joint Branch Planning...")
    print("-" * 70)
    joint_branch = JointBranch(config, run_state_dict)
    joint_branch.run_planning(python_ref)

    # Print results
    print("\n" + "=" * 70)
    print("Joint Branch Planning Results")
    print("=" * 70)

    print("\n--- Joint Plan ---")
    joint_plan = run_state_dict.joint_plan
    if joint_plan:
        print(f"  tiling_strategy: {joint_plan.get('tiling_strategy')}")
        print(f"  tiling_fields: {joint_plan.get('tiling_fields')}")
        print(f"\n  kernel_pseudocode:\n{joint_plan.get('kernel_pseudocode', '(none)')[:500]}")
        print(f"\n  tiling_execution:\n{joint_plan.get('tiling_execution', '(none)')[:500]}")
        print(f"\n  retrieval_requests: {joint_plan.get('retrieval_requests', [])}")

    print("\n--- Knowledge Context (first 500 chars) ---")
    if run_state_dict.knowledge_context:
        ctx = run_state_dict.knowledge_context
        print(ctx[:500] + "..." if len(ctx) > 500 else ctx)
    else:
        print("(none)")

    # Output context for _config.py
    print("\n" + "=" * 70)
    print(f"Copy this to _config.py:")
    print("=" * 70)
    print(f'''
JOINT_PLAN_CONTEXT["{test_case}"] = {repr(joint_plan)}

KNOWLEDGE_CONTEXT["{test_case}"] = """{run_state_dict.knowledge_context[:2000] if run_state_dict.knowledge_context else ''}"""
''')


if __name__ == "__main__":
    test_case = sys.argv[1] if len(sys.argv) > 1 else "hard"
    main(test_case)
