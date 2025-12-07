# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Ascend C code templates for operator generation.

This module provides template generation for the 6 components of an Ascend C operator:
1. project_json_src - Operator project configuration
2. host_tiling_src - Tiling data structure definition
3. host_operator_src - Host-side operator implementation
4. kernel_src - Device kernel (provided by LLM)
5. python_bind_src - Python binding via pybind11
6. model_src - Test model for verification

Only kernel_src needs to be provided by LLM, others are auto-generated.
"""

from .generator import AscendCTemplateGenerator

__all__ = ["AscendCTemplateGenerator"]
