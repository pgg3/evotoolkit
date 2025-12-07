# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Template Generator

Demonstrates two tiling modes:
1. Default mode - only kernel_src (for element-wise operators)
2. Full LLM mode - kernel_src + host_tiling_src + host_operator_src (for complex operators)

Usage:
    python 2_template_generator.py
"""

from evotoolkit.task.cann_init import (
    OperatorSignatureParser,
    AscendCTemplateGenerator,
)
from _config import (
    PYTHON_REFERENCE,
    KERNEL_SRC,
    BLOCK_DIM,
    HOST_TILING_SRC,
    HOST_OPERATOR_SRC,
)


def test_default_mode(gen):
    """Test default tiling mode (element-wise operators)."""
    print("\n" + "-" * 40)
    print("Mode 1: Default (element-wise)")
    print("-" * 40)

    full_code = gen.generate(kernel_src=KERNEL_SRC, block_dim=BLOCK_DIM)

    print("Generated components:")
    for key, value in full_code.items():
        print(f"  {key}: {value.count(chr(10)) + 1} lines")

    print("\nhost_tiling_src (auto-generated):")
    print(full_code["host_tiling_src"])


def test_full_llm_mode(gen):
    """Test full LLM mode with complete host code."""
    print("\n" + "-" * 40)
    print("Mode 2: Full LLM (complete host code)")
    print("-" * 40)

    full_code = gen.generate(
        kernel_src=KERNEL_SRC,
        host_tiling_src=HOST_TILING_SRC,
        host_operator_src=HOST_OPERATOR_SRC,
    )

    print("Generated components:")
    for key, value in full_code.items():
        print(f"  {key}: {value.count(chr(10)) + 1} lines")

    print("\nhost_tiling_src (LLM provided):")
    print(full_code["host_tiling_src"][:300] + "...")


def main():
    print("=" * 50)
    print("Template Generator Test")
    print("=" * 50)

    # Parse signature
    parser = OperatorSignatureParser()
    signature = parser.parse(PYTHON_REFERENCE, "add")

    # Create generator
    gen = AscendCTemplateGenerator(signature)

    # Test both modes
    test_default_mode(gen)
    test_full_llm_mode(gen)


if __name__ == "__main__":
    main()
