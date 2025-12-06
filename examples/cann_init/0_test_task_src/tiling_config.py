# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Tiling configuration for Add operator.

This file demonstrates the tiling information that LLM needs to generate
for complex operators. For simple element-wise operators, the default
tiling logic in AscendCTemplateGenerator can be used instead.

Two modes supported:
1. Default mode: Only provide kernel_src, use default tiling for element-wise ops
2. Custom mode: Provide kernel_src + tiling_fields + tiling_func_body for complex ops
"""

# ============================================================================
# Tiling Fields Definition
# ============================================================================
# These define the TilingData struct fields that are passed from host to device.
# The kernel code accesses these via GET_TILING_DATA macro.
#
# For Add operator, we need:
# - totalLength: total number of elements to process
# - tileNum: number of tiles (usually equals BLOCK_DIM)

TILING_FIELDS = [
    {"name": "totalLength", "type": "uint32_t"},
    {"name": "tileNum", "type": "uint32_t"},
]


# ============================================================================
# TilingFunc Body
# ============================================================================
# This is the body of TilingFunc that runs on host to compute tiling parameters.
# It calculates values for all fields defined in TILING_FIELDS.
#
# Available context APIs:
# - context->GetInputShape(i) - get i-th input tensor shape
# - context->GetInputTensor(i) - get i-th input tensor
# - context->GetOutputShape(i) - get i-th output tensor shape
#
# Must set tiling fields via tiling.set_XXX() methods.

TILING_FUNC_BODY = """
    // Get input tensor shape
    auto shape = context->GetInputShape(0)->GetStorageShape();

    // Calculate total number of elements
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalLength *= shape.GetDim(i);
    }

    // Set tiling parameters
    // For element-wise ops, each core processes totalLength/BLOCK_DIM elements
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(BLOCK_DIM);
"""


# ============================================================================
# Block Dimension
# ============================================================================
# Number of AI Cores to use for parallel execution.
# Typical values: 8, 16, 32 depending on NPU model.

BLOCK_DIM = 8


# ============================================================================
# Complete LLM Output Structure (for reference)
# ============================================================================
# This is what the Coder Agent should output in the agentic framework:
#
# {
#     "kernel_src": "...",           # The device kernel code
#     "tiling_fields": [...],        # Tiling data fields
#     "tiling_func_body": "...",     # TilingFunc implementation
#     "block_dim": 8                 # Number of parallel cores
# }

def get_llm_output_example():
    """Return an example of complete LLM output structure."""
    from pathlib import Path
    kernel_src = (Path(__file__).parent / "kernel_src.cpp").read_text()

    return {
        "kernel_src": kernel_src,
        "tiling_fields": TILING_FIELDS,
        "tiling_func_body": TILING_FUNC_BODY,
        "block_dim": BLOCK_DIM,
    }
