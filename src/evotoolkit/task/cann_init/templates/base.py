# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Base utilities and type conversion for Ascend C template generation.
"""

from typing import Any, Dict, List


class TemplateBase:
    """Base class with common utilities for template generation."""

    def __init__(self, signature: Dict[str, Any]):
        """
        Initialize with operator signature.

        Args:
            signature: Operator signature containing:
                - op_name: Operator name (e.g., "add")
                - inputs: List of input info [{name, dtype, is_tensor}]
                - outputs: List of output info [{name, dtype, is_tensor}]
                - init_params: List of __init__ param info [{name, dtype, is_tensor, default}]
        """
        self.signature = signature
        self.op_name = signature["op_name"]
        self.op_name_lower = self.op_name.lower()
        self.op_name_capital = self._to_pascal_case(self.op_name)
        self.op_custom = f"{self.op_name_lower}_custom"
        self.op_custom_capital = self._to_pascal_case(self.op_custom)

    def _to_pascal_case(self, name: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))

    def _to_camel_case(self, name: str) -> str:
        """Convert snake_case to camelCase."""
        parts = name.split("_")
        return parts[0] + "".join(word.capitalize() for word in parts[1:])

    def _collect_scalar_params(self) -> List[Dict[str, Any]]:
        """Collect all scalar (non-tensor) parameters from inputs and init_params."""
        scalar_params = []

        for inp in self.signature.get("inputs", []):
            if not inp.get("is_tensor", True):
                scalar_params.append(inp)

        for param in self.signature.get("init_params", []):
            if not param.get("is_tensor", False):
                scalar_params.append(param)

        return scalar_params

    def _dtype_to_cpp_type(self, dtype: str) -> str:
        """Convert Python dtype to C++ type (for scalar parameters in pybind)."""
        dtype_map = {
            "float": "float",
            "float32": "float",
            "float16": "float",  # Use float for API, cast internally
            "int": "int64_t",
            "int32": "int32_t",
            "int64": "int64_t",
            "bool": "bool",
        }
        return dtype_map.get(dtype.lower(), "float")

    def _dtype_to_cann_json(self, dtype: str) -> str:
        """Convert Python dtype to CANN JSON type string (for tensor types)."""
        dtype_map = {
            "float": "float",
            "float32": "float",  # CANN uses "float" not "float32"
            "float16": "float16",
            "bfloat16": "bfloat16",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "bool": "bool",
        }
        return dtype_map.get(dtype.lower(), "float")

    def _dtype_to_ge_datatype(self, dtype: str) -> str:
        """Convert Python dtype to ge::DataType enum (for tensor types)."""
        dtype_map = {
            "float": "ge::DT_FLOAT",
            "float32": "ge::DT_FLOAT",
            "float16": "ge::DT_FLOAT16",
            "bfloat16": "ge::DT_BF16",
            "int8": "ge::DT_INT8",
            "int16": "ge::DT_INT16",
            "int32": "ge::DT_INT32",
            "int64": "ge::DT_INT64",
            "uint8": "ge::DT_UINT8",
            "bool": "ge::DT_BOOL",
        }
        return dtype_map.get(dtype.lower(), "ge::DT_FLOAT")
