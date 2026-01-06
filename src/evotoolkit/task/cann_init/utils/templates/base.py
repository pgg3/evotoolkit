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
        """Convert Python dtype to CANN JSON type string (for tensor types).

        Maps Python/PyTorch dtype to CANN msopgen JSON format type.
        Reference: /usr/local/Ascend/.../op_gen/config/transform.json INPUT_OUTPUT_DTYPE_MAP
        """
        dtype_map = {
            # Float types
            "float": "float",
            "float32": "float",  # CANN uses "float" not "float32"
            "fp32": "float",     # Alias
            "half": "float16",   # PyTorch alias
            "fp16": "float16",   # Alias
            "float16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",  # Alias
            "double": "double",
            "float64": "double",
            # Integer types
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int": "int32",      # Python int -> int32 by default
            "int64": "int64",
            "long": "int64",     # PyTorch alias
            # Unsigned integer types
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "uint64": "uint64",
            # Other types
            "bool": "bool",
            "complex64": "complex64",
            "complex128": "complex128",
        }
        return dtype_map.get(dtype.lower(), "float")

    def _dtype_to_cann_attr_type(self, dtype: str) -> str:
        """Convert Python dtype to CANN JSON attr type string (for scalar attrs).

        Reference: /usr/local/Ascend/.../op_gen/config/transform.json IR_ATTR_TYPE_MAP
        """
        # Already in CANN format - pass through directly
        cann_attr_types = {
            "int", "float", "bool", "string", "type",
            "list_int", "list_float", "list_bool", "list_string",
            "list_list_int", "tensor", "list_tensor", "list_type",
        }
        dtype_lower = dtype.lower()
        if dtype_lower in cann_attr_types:
            return dtype_lower

        # Python type to CANN attr type
        dtype_map = {
            "int": "int",
            "int32": "int",
            "int64": "int",
            "float": "float",
            "float32": "float",
            "float16": "float",
            "double": "float",
            "bool": "bool",
            "str": "string",
        }
        return dtype_map.get(dtype_lower, "float")

    def _dtype_to_ge_datatype(self, dtype: str) -> str:
        """Convert Python dtype to ge::DataType enum (for tensor types).

        Reference: /usr/local/Ascend/.../include/graph/types.h enum DataType
        """
        dtype_map = {
            # Float types
            "float": "ge::DT_FLOAT",
            "float32": "ge::DT_FLOAT",
            "fp32": "ge::DT_FLOAT",
            "half": "ge::DT_FLOAT16",
            "fp16": "ge::DT_FLOAT16",
            "float16": "ge::DT_FLOAT16",
            "bfloat16": "ge::DT_BF16",
            "bf16": "ge::DT_BF16",
            "double": "ge::DT_DOUBLE",
            "float64": "ge::DT_DOUBLE",
            # Integer types
            "int8": "ge::DT_INT8",
            "int16": "ge::DT_INT16",
            "int32": "ge::DT_INT32",
            "int": "ge::DT_INT32",
            "int64": "ge::DT_INT64",
            "long": "ge::DT_INT64",
            # Unsigned integer types
            "uint8": "ge::DT_UINT8",
            "uint16": "ge::DT_UINT16",
            "uint32": "ge::DT_UINT32",
            "uint64": "ge::DT_UINT64",
            # Other types
            "bool": "ge::DT_BOOL",
            "complex64": "ge::DT_COMPLEX64",
            "complex128": "ge::DT_COMPLEX128",
        }
        return dtype_map.get(dtype.lower(), "ge::DT_FLOAT")
