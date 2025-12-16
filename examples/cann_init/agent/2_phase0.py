#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Phase 0 Analyzer 独立测试

测试 Phase 0 的两个功能:
1. 签名解析 (Signature Parsing)
2. 形状分析与策略决策 (Shape Analysis & Strategy Decision)

输入: op_name, python_ref
输出: signature, shape_inference, strategies, etc.

用法:
    python 2_phase0.py [easy|medium|hard]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_llm, get_test_config, load_python_ref, ensure_output_dir
)

from evotoolkit.task.cann_init import CANNIniterInterface, CANNInitTask
from evotoolkit.evo_method.cann_initer import CANNIniterConfig
from evotoolkit.evo_method.cann_initer.phases import Phase0Analyzer
from evotoolkit.evo_method.cann_initer.run_state_dict import CANNIniterRunStateDict


def main(test_case: str = "hard"):
    config_info = get_test_config(test_case)
    op_name = config_info["op_name"]
    python_ref = load_python_ref(test_case)

    print("=" * 70)
    print(f"Phase 0 Analyzer Test - {config_info['name']}")
    print("=" * 70)

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
        output_path=str(ensure_output_dir(f"phase0_{test_case}")),
        running_llm=llm,
        knowledge_base=None,
        verbose=True,
    )
    run_state_dict = CANNIniterRunStateDict()

    # Run Phase 0
    print("\n[2] Running Phase 0 Analyzer...")
    print("-" * 70)
    phase0 = Phase0Analyzer(config, run_state_dict)
    phase0.analyze(op_name, python_ref)

    # Print results
    print("\n" + "=" * 70)
    print("Phase 0 Results")
    print("=" * 70)

    print("\n--- Signature ---")
    print(run_state_dict.signature)

    print("\n--- Shape Inference ---")
    print(f"  {run_state_dict.shape_inference}")

    print("\n--- Functionality ---")
    print(f"  {run_state_dict.functionality}")

    print("\n--- Strategies ---")
    for key, value in run_state_dict.strategies.items():
        print(f"  - {key}: {value}")

    # Output context for _config.py
    print("\n" + "=" * 70)
    print(f"Copy this to _config.py PHASE0_CONTEXT['{test_case}']:")
    print("=" * 70)
    print(f'''
    "{test_case}": {{
        "op_name": "{op_name}",
        "signature": {repr(run_state_dict.signature)},
        "shape_inference": {repr(run_state_dict.shape_inference)},
        "functionality": {repr(run_state_dict.functionality)},
        "strategies": {repr(run_state_dict.strategies)},
    }},
''')


if __name__ == "__main__":
    test_case = sys.argv[1] if len(sys.argv) > 1 else "hard"
    main(test_case)
