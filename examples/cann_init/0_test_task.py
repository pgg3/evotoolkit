# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Basic CANN Init Task Test Example

This example demonstrates how to create and test a CANNInitTask
for Ascend C operator generation.

Requirements:
- pip install evotoolkit
- Ascend NPU environment (for real evaluation)
- CANN toolkit installed

Usage:
    # Test with fake mode (no NPU required, default)
    python 0_test_task.py
    python 0_test_task.py --fake

    # Test with real NPU evaluation
    python 0_test_task.py --real

    # Test with specific NPU type
    python 0_test_task.py --real --npu Ascend910B

    # Test with real NPU and cleanup after
    python 0_test_task.py --real --cleanup
"""

import argparse
import sys
from pathlib import Path

from evotoolkit.task.cann_init import (
    CANNInitTask,
    AscendCTemplateGenerator,
    OperatorSignatureParser,
)
from evotoolkit.core import Solution


# ============================================================================
# Load source files from 0_test_task_src/
# ============================================================================
SRC_DIR = Path(__file__).parent / "0_test_task_src"

PYTHON_REFERENCE = (SRC_DIR / "python_reference.py").read_text()
KERNEL_SRC = (SRC_DIR / "kernel_src.cpp").read_text()

# Load tiling configuration (for custom tiling mode)
# Using exec to load from directory with numeric prefix
_tiling_config = {}
exec((SRC_DIR / "tiling_config.py").read_text(), _tiling_config)
TILING_FIELDS = _tiling_config["TILING_FIELDS"]
TILING_FUNC_BODY = _tiling_config["TILING_FUNC_BODY"]
BLOCK_DIM = _tiling_config["BLOCK_DIM"]


def test_signature_parser():
    """Test the signature parser."""
    print("\n" + "=" * 60)
    print("Test 1: Signature Parser")
    print("=" * 60)

    parser = OperatorSignatureParser()
    signature = parser.parse(PYTHON_REFERENCE, "add")

    print(f"Op Name: {signature['op_name']}")
    print(f"Inputs: {signature['inputs']}")
    print(f"Outputs: {signature['outputs']}")
    print(f"Init Params: {signature['init_params']}")
    print(f"Dtypes: {signature['dtypes']}")

    return signature


def test_template_generator(signature):
    """Test the template generator with DEFAULT tiling mode."""
    print("\n" + "=" * 60)
    print("Test 2a: Template Generator (Default Tiling Mode)")
    print("=" * 60)
    print("Mode: Only kernel_src provided, using default tiling logic")
    print("Use case: Simple element-wise operators (Add, Mul, etc.)")

    gen = AscendCTemplateGenerator(signature)

    # Generate with default settings (no custom tiling)
    full_code = gen.generate(
        kernel_src=KERNEL_SRC,
        block_dim=8,
    )

    print("\nGenerated components:")
    for key, value in full_code.items():
        lines = value.count('\n') + 1
        print(f"  - {key}: {lines} lines")

    # Show a snippet of each
    print("\n--- project_json_src (snippet) ---")
    print(full_code["project_json_src"][:200])

    print("\n--- host_tiling_src (snippet) ---")
    print(full_code["host_tiling_src"][:300])

    return full_code


def test_template_generator_custom_tiling(signature):
    """Test the template generator with CUSTOM tiling mode."""
    print("\n" + "=" * 60)
    print("Test 2b: Template Generator (Custom Tiling Mode)")
    print("=" * 60)
    print("Mode: Full LLM output with custom tiling_fields and tiling_func_body")
    print("Use case: Complex operators (Softmax, LayerNorm, Reduction, etc.)")

    gen = AscendCTemplateGenerator(signature)

    # Generate with custom tiling (simulating full LLM output)
    full_code = gen.generate(
        kernel_src=KERNEL_SRC,
        block_dim=BLOCK_DIM,
        tiling_fields=TILING_FIELDS,
        tiling_func_body=TILING_FUNC_BODY,
    )

    print("\nGenerated components:")
    for key, value in full_code.items():
        lines = value.count('\n') + 1
        print(f"  - {key}: {lines} lines")

    # Show tiling-related code to verify custom tiling was used
    print("\n--- host_tiling_src (full) ---")
    print(full_code["host_tiling_src"])

    print("\n--- host_operator_src TilingFunc (snippet) ---")
    # Extract TilingFunc part
    op_src = full_code["host_operator_src"]
    tiling_start = op_src.find("static ge::graphStatus TilingFunc")
    tiling_end = op_src.find("}", tiling_start + 100) + 1
    print(op_src[tiling_start:tiling_end + 50])

    return full_code


def test_task_fake_mode():
    """Test CANNInitTask in fake mode."""
    print("\n" + "=" * 60)
    print("Test 3: CANNInitTask (fake_mode=True)")
    print("=" * 60)

    task = CANNInitTask(
        data={
            "op_name": "add",
            "python_reference": PYTHON_REFERENCE,
            "npu_type": "Ascend910B",
        },
        fake_mode=True,
    )

    print(f"Task Type: {task.get_task_type()}")
    print(f"Op Name: {task.op_name}")
    print(f"NPU Type: {task.npu_type}")

    # Test evaluate_code
    print("\n--- Testing evaluate_code() ---")
    result = task.evaluate_code(KERNEL_SRC)
    print(f"Valid: {result.valid}")
    print(f"Score: {result.score}")
    print(f"Additional Info: {result.additional_info}")

    # Test evaluate_solution
    print("\n--- Testing evaluate_solution() ---")
    solution = Solution(
        sol_string=KERNEL_SRC,
        other_info={"block_dim": 16},
    )
    result = task.evaluate_solution(solution)
    print(f"Valid: {result.valid}")
    print(f"Score: {result.score}")

    return task


def test_task_real_mode(npu_type: str = "Ascend910B", cleanup: bool = False):
    """Test CANNInitTask with real NPU evaluation."""
    print("\n" + "=" * 60)
    print("Test 4: CANNInitTask (Real NPU Evaluation)")
    print("=" * 60)

    # Use directory next to this test file instead of temp dir
    project_path = Path(__file__).parent / "0_test_task_project"
    project_path.mkdir(exist_ok=True)
    print(f"Project path: {project_path}")
    print(f"NPU type: {npu_type}")

    task = CANNInitTask(
        data={
            "op_name": "add",
            "python_reference": PYTHON_REFERENCE,
            "npu_type": npu_type,
        },
        project_path=str(project_path),
        fake_mode=False,
    )

    print(f"Task Type: {task.get_task_type()}")
    print("\n--- Evaluating kernel (this may take a few minutes) ---")

    result = task.evaluate_code(KERNEL_SRC)

    print(f"\nResult:")
    print(f"  Valid: {result.valid}")
    print(f"  Score: {result.score}")
    print(f"  Stage: {result.additional_info.get('stage')}")

    if result.valid:
        print(f"  Runtime: {result.additional_info.get('runtime'):.4f} ms")
    else:
        print(f"  Error: {result.additional_info.get('error')}")

    if cleanup:
        print("\nCleaning up generated files...")
        task.cleanup()
    else:
        print(f"\nGenerated files kept at: {project_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test CANNInitTask")
    parser.add_argument(
        "--fake", action="store_true",
        help="Run in fake mode (no NPU required)"
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Run real NPU evaluation"
    )
    parser.add_argument(
        "--npu", type=str, default="Ascend910B",
        help="NPU type (default: Ascend910B)"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Cleanup generated project after test (default: keep files)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CANN Init Task Test")
    print("=" * 60)

    # Test 1: Signature Parser
    signature = test_signature_parser()

    # Test 2a: Template Generator - Default Tiling Mode
    # Use case: Simple element-wise operators where tiling logic is standard
    test_template_generator(signature)

    # Test 2b: Template Generator - Custom Tiling Mode
    # Use case: Complex operators where LLM generates custom tiling logic
    test_template_generator_custom_tiling(signature)

    # Test 3: Task in fake mode
    if args.fake or not args.real:
        test_task_fake_mode()

    # Test 4: Real NPU evaluation (optional)
    if args.real:
        try:
            test_task_real_mode(npu_type=args.npu, cleanup=args.cleanup)
        except Exception as e:
            print(f"\nReal mode test failed: {e}")
            print("Make sure you're running on an Ascend NPU environment.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
