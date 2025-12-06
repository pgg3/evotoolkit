# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Template Generator

Demonstrates three tiling modes:
1. Default mode - only kernel_src
2. Template mode - kernel_src + tiling_fields + tiling_func_body
3. Direct mode - kernel_src + host_tiling_src + host_operator_src

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
    TILING_FIELDS,
    TILING_FUNC_BODY,
    BLOCK_DIM,
)


def test_default_mode(gen):
    """Test default tiling mode (element-wise operators)."""
    print("\n" + "-" * 40)
    print("Mode 1: Default (element-wise)")
    print("-" * 40)

    full_code = gen.generate(kernel_src=KERNEL_SRC)

    print("Generated components:")
    for key, value in full_code.items():
        print(f"  {key}: {value.count(chr(10)) + 1} lines")


def test_template_mode(gen):
    """Test template mode with custom tiling params."""
    print("\n" + "-" * 40)
    print("Mode 2: Template (custom tiling)")
    print("-" * 40)

    full_code = gen.generate(
        kernel_src=KERNEL_SRC,
        block_dim=BLOCK_DIM,
        tiling_fields=TILING_FIELDS,
        tiling_func_body=TILING_FUNC_BODY,
    )

    print("Generated components:")
    for key, value in full_code.items():
        print(f"  {key}: {value.count(chr(10)) + 1} lines")

    print("\nhost_tiling_src:")
    print(full_code["host_tiling_src"])


def test_direct_mode(gen):
    """Test direct mode with complete host code."""
    print("\n" + "-" * 40)
    print("Mode 3: Direct (LLM generates complete host code)")
    print("-" * 40)

    # Simulate LLM-generated complete host code
    host_tiling_src = """#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}
#endif
"""

    full_code = gen.generate(
        kernel_src=KERNEL_SRC,
        host_tiling_src=host_tiling_src,
        # host_operator_src not provided, will use template
    )

    print("Generated components:")
    for key, value in full_code.items():
        print(f"  {key}: {value.count(chr(10)) + 1} lines")

    print("\nhost_tiling_src (direct):")
    print(full_code["host_tiling_src"])


def main():
    print("=" * 50)
    print("Template Generator Test")
    print("=" * 50)

    # Parse signature
    parser = OperatorSignatureParser()
    signature = parser.parse(PYTHON_REFERENCE, "add")

    # Create generator
    gen = AscendCTemplateGenerator(signature)

    # Test all modes
    test_default_mode(gen)
    test_template_mode(gen)
    test_direct_mode(gen)


if __name__ == "__main__":
    main()
