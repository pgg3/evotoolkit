# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Component 1: project_json_src generator.

Generates the operator project configuration JSON for msopgen.
"""

import json
from typing import Any, Dict

from .base import TemplateBase


class ProjectJsonGenerator(TemplateBase):
    """Generate project JSON configuration for Ascend C operator."""

    def generate(self) -> str:
        """
        Generate project JSON configuration.

        Returns:
            JSON string for msopgen project configuration.

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
                input_desc.append({
                    "name": inp["name"],
                    "param_type": "required",
                    "format": ["ND"],
                    "type": [cann_type],
                })
            else:
                # Scalar input as attr
                attr_info = {
                    "name": inp["name"],
                    "param_type": "required",
                    "type": inp.get("dtype", "float"),
                }
                attr_desc.append(attr_info)

        # Process init_params
        for param in init_params:
            if param.get("is_tensor", False):
                cann_type = self._dtype_to_cann_json(param.get("dtype", "float"))
                input_desc.append({
                    "name": param["name"],
                    "param_type": "required",
                    "format": ["ND"],
                    "type": [cann_type],
                })
            else:
                # Scalar as attr
                attr_info = {
                    "name": param["name"],
                    "type": param.get("dtype", "float"),
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
                output_desc.append({
                    "name": out["name"],
                    "param_type": "required",
                    "format": ["ND"],
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
