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

Code has been refactored into the templates/ subpackage for better organization:
- templates/base.py          - Base utilities and type conversion
- templates/project_json.py  - Component 1: project JSON configuration
- templates/host_tiling.py   - Component 2: tiling data structure
- templates/host_operator.py - Component 3: host-side operator
- templates/python_bind.py   - Component 5: Python binding
- templates/moddel_src.py     - Component 6: test model
- templates/generator.py     - Main orchestrator class
"""

# Re-export the main class for backward compatibility
from .templates import AscendCTemplateGenerator

__all__ = ["AscendCTemplateGenerator"]
