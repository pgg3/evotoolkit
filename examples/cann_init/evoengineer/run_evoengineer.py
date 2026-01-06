# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
EvoEngineer CANN Kernel Generation Test

Usage:
    python run_evoengineer.py                    # Run with default settings
    python run_evoengineer.py --max-samples 10   # Limit samples
    python run_evoengineer.py --dry-run          # Test prompt generation only
"""

import argparse
import os

from evotoolkit.task.cann_init import CANNInitTask, EvoEngineerCANNInterface
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig
from evotoolkit.tools.llm import HttpsApi

from _config import get_task_data, ensure_output_dir


def create_llm():
    """Create LLM client from environment variables"""
    api_url = os.environ.get("LLM_API_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "gpt-4o")

    if not api_url or not api_key:
        raise ValueError(
            "Please set LLM_API_URL and LLM_API_KEY environment variables.\n"
            "Example:\n"
            "  export LLM_API_URL=https://api.openai.com/v1/chat/completions\n"
            "  export LLM_API_KEY=sk-xxx"
        )

    return HttpsApi(api_url=api_url, key=api_key, model=model, timeout=120)


def test_prompt_generation(interface: EvoEngineerCANNInterface):
    """Test prompt generation without running LLM"""
    print("\n" + "=" * 50)
    print("Testing Prompt Generation")
    print("=" * 50)

    # Test init prompt
    init_sol = interface.make_init_sol()
    prompt = interface.get_operator_prompt("init", [], init_sol, [])
    print("\n--- Init Prompt ---")
    print(prompt[0]["content"][:1000] + "...")

    # Test parse_response
    mock_response = """name: test_kernel
thought: Testing the parser

kernel_src:
```cpp
#include "kernel_operator.h"
void test() {}
```

tiling_fields:
```json
[{"name": "totalLength", "type": "uint32_t"}]
```

tiling_func_body:
```cpp
return ge::GRAPH_SUCCESS;
```

infer_shape_body:
```cpp
return GRAPH_SUCCESS;
```"""

    parsed = interface.parse_response(mock_response)
    print("\n--- Parsed Response ---")
    print(f"Name: {parsed.other_info.get('name')}")
    print(f"Kernel src length: {len(parsed.sol_string)}")
    print(f"Tiling fields: {parsed.other_info.get('tiling_fields')}")


def main():
    parser = argparse.ArgumentParser(description="EvoEngineer CANN Test")
    parser.add_argument("--npu", default="Ascend910B", help="NPU type")
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples")
    parser.add_argument("--max-generations", type=int, default=3, help="Max generations")
    parser.add_argument("--pop-size", type=int, default=3, help="Population size")
    parser.add_argument("--num-samplers", type=int, default=2, help="Parallel samplers")
    parser.add_argument("--dry-run", action="store_true", help="Test prompt only")
    args = parser.parse_args()

    print("=" * 50)
    print("EvoEngineer CANN Kernel Generation")
    print("=" * 50)

    # Create task
    task = CANNInitTask(
        data=get_task_data(npu_type=args.npu),
        fake_mode=False,
    )
    print(f"Task: {task.get_task_type()}")
    print(f"Op Name: {task.op_name}")

    # Create interface
    interface = EvoEngineerCANNInterface(task)

    # Dry run mode - just test prompt generation
    if args.dry_run:
        test_prompt_generation(interface)
        return

    # Create LLM client
    llm = create_llm()
    print(f"LLM Model: {llm._model}")

    # Create output directory
    output_dir = ensure_output_dir("evoengineer_run")

    # Create config
    config = EvoEngineerConfig(
        interface=interface,
        output_path=str(output_dir),
        running_llm=llm,
        verbose=True,
        max_generations=args.max_generations,
        max_sample_nums=args.max_samples,
        pop_size=args.pop_size,
        num_samplers=args.num_samplers,
        num_evaluators=2,
    )

    # Run EvoEngineer
    print("\nStarting EvoEngineer...")
    evo = EvoEngineer(config)
    evo.run()

    # Print results
    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)

    valid_sols = [
        s for s in evo.run_state_dict.sol_history
        if s.evaluation_res and s.evaluation_res.valid
    ]

    print(f"Total samples: {evo.run_state_dict.tot_sample_nums}")
    print(f"Valid solutions: {len(valid_sols)}")

    if valid_sols:
        best = max(valid_sols, key=lambda x: x.evaluation_res.score)
        print(f"\nBest solution:")
        print(f"  Name: {best.other_info.get('name')}")
        print(f"  Runtime: {-best.evaluation_res.score:.4f} ms")
        print(f"  Thought: {best.other_info.get('thought', '')[:100]}...")


if __name__ == "__main__":
    main()
