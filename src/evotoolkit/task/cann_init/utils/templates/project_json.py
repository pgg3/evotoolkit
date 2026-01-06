# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Component 1: project_json_src generator.

Generates the operator project configuration JSON for msopgen.
"""

import json
from typing import Any, Dict

from .base import TemplateBase

# Default format for tensors
DEFAULT_FORMAT = "ND"


class ProjectJsonGenerator(TemplateBase):
    """Generate project JSON configuration for Ascend C operator."""

    def generate(self) -> str:
        """
        Generate project JSON configuration.

        Returns:
            JSON string for msopgen project configuration.

        Signature format支持:
            inputs/outputs/init_params 中的每个元素可以包含:
            - name: 参数名 (必需)
            - dtype: 数据类型 (默认 "float")
            - is_tensor: 是否为tensor (默认 True for inputs/outputs, False for init_params)
            - format: tensor格式 (可选, 默认 "ND", 可选 "NCHW")
            - default: 默认值 (仅用于 init_params)

        Example output:
        ```json
        [{
            "op": "AddCustom",
            "language": "cpp",
            "input_desc": [
                {"name": "x", "param_type": "required", "format": ["ND"], "type": ["float"]},
                {"name": "y", "param_type": "required", "format": ["ND"], "type": ["float"]}
            ],
            "output_desc": [
                {"name": "z", "param_type": "required", "format": ["ND"], "type": ["float"]}
            ]
        }]
        ```
        """
        inputs = self.signature.get("inputs", [])
        outputs = self.signature.get("outputs", [])
        init_params = self.signature.get("init_params", [])

        input_desc = []
        attr_desc = []

        # Process all inputs
        for inp in inputs:
            if inp.get("is_tensor", True):
                cann_type = self._dtype_to_cann_json(inp.get("dtype", "float"))
                fmt = inp.get("format", DEFAULT_FORMAT)
                input_desc.append({
                    "name": inp["name"],
                    "param_type": "required",
                    "format": [fmt],
                    "type": [cann_type],
                })
            else:
                # Scalar input as attr
                attr_type = self._dtype_to_cann_attr_type(inp.get("dtype", "float"))
                attr_info = {
                    "name": inp["name"],
                    "param_type": "required",
                    "type": attr_type,
                }
                attr_desc.append(attr_info)

        # Process init_params
        for param in init_params:
            if param.get("is_tensor", False):
                cann_type = self._dtype_to_cann_json(param.get("dtype", "float"))
                fmt = param.get("format", DEFAULT_FORMAT)
                input_desc.append({
                    "name": param["name"],
                    "param_type": "required",
                    "format": [fmt],
                    "type": [cann_type],
                })
            else:
                # Scalar as attr
                attr_type = self._dtype_to_cann_attr_type(param.get("dtype", "float"))
                attr_info = {
                    "name": param["name"],
                    "type": attr_type,
                }
                # Check if optional (has default value)
                if "default" in param and param["default"] is not None:
                    attr_info["param_type"] = "optional"
                    attr_info["default_value"] = str(param["default"])
                else:
                    attr_info["param_type"] = "required"
                attr_desc.append(attr_info)

        output_desc = []
        for out in outputs:
            if out.get("is_tensor", True):
                cann_type = self._dtype_to_cann_json(out.get("dtype", "float"))
                fmt = out.get("format", DEFAULT_FORMAT)
                output_desc.append({
                    "name": out["name"],
                    "param_type": "required",
                    "format": [fmt],
                    "type": [cann_type],
                })

        config = [{
            "op": self.op_custom_capital,
            "language": "cpp",
            "input_desc": input_desc,
            "output_desc": output_desc,
        }]

        # Add attr if any scalar params exist
        if attr_desc:
            config[0]["attr"] = attr_desc

        return json.dumps(config, indent=4)
