#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Pybind Branch 独立测试

测试 Pybind 分支的代码生成 (与 Joint Branch 并行):
- 根据 signature 生成 pybind 绑定代码

输入: signature (来自 Phase 0)
输出: pybind_src

用法:
    python 3_pybind.py [easy|medium|hard]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_llm, get_test_config, load_python_ref, ensure_output_dir,
    get_phase0_context
)

from evotoolkit.task.cann_init import CANNIniterInterface, CANNInitTask
from evotoolkit.evo_method.cann_initer import CANNIniterConfig
from evotoolkit.evo_method.cann_initer.phases import PybindBranch
from evotoolkit.evo_method.cann_initer.run_state_dict import CANNIniterRunStateDict


def main(test_case: str = "hard"):
    config_info = get_test_config(test_case)
    python_ref = load_python_ref(test_case)

    print("=" * 70)
    print(f"Pybind Branch Test - {config_info['name']}")
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

    config = CANNIniterConfig(
        task=task,
        interface=interface,
        output_path=str(ensure_output_dir(f"pybind_{test_case}")),
        running_llm=llm,
        knowledge_base=None,
        verbose=True,
    )

    # Restore Phase 0 state
    run_state_dict = CANNIniterRunStateDict()
    run_state_dict.op_name = phase0_ctx["op_name"]
    run_state_dict.signature = phase0_ctx["signature"]
    run_state_dict.strategies = phase0_ctx["strategies"]

    # Run Pybind Branch
    print("\n[2] Running Pybind Branch...")
    print("-" * 70)
    pybind_branch = PybindBranch(config, run_state_dict)
    pybind_branch.run()

    # Print results
    print("\n" + "=" * 70)
    print("Pybind Branch Results")
    print("=" * 70)

    print("\n--- Generated pybind_src ---")
    if run_state_dict.pybind_src:
        print(run_state_dict.pybind_src)
    else:
        print("(None - using default template)")

    # Save output to both pybind_xxx and impl_xxx directories
    output_dir = ensure_output_dir(f"pybind_{test_case}")
    impl_dir = ensure_output_dir(f"impl_{test_case}")

    if run_state_dict.pybind_src:
        (output_dir / "pybind_src.cpp").write_text(run_state_dict.pybind_src)
        (impl_dir / "pybind_src.cpp").write_text(run_state_dict.pybind_src)
        print(f"\n[Saved to {output_dir / 'pybind_src.cpp'}]")
        print(f"[Saved to {impl_dir / 'pybind_src.cpp'}]")


if __name__ == "__main__":
    test_case = sys.argv[1] if len(sys.argv) > 1 else "hard"
    main(test_case)
