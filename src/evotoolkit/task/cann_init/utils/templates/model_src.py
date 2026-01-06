# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Component 6: model_src generator.

Generates test model code (ModelNew class) for verification.
"""

from .base import TemplateBase


class ModelSrcGenerator(TemplateBase):
    """Generate test model code for Ascend C operator."""

    def generate(self, project_path: str) -> str:
        """
        Generate test model code (ModelNew class).

        ModelNew must have the same interface as Model:
        - Same __init__ parameters
        - Same forward parameters

        Args:
            project_path: Absolute path to project directory (for .so loading)

        Returns:
            Complete Python model file content.

        Example output (simple case):
        ```python
        import torch
        import torch_npu
        import custom_ops_lib

        class ModelNew(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return custom_ops_lib.add_custom(x, y)
        ```

        Example output (with init params):
        ```python
        import torch
        import torch_npu
        import custom_ops_lib

        class ModelNew(torch.nn.Module):
            def __init__(self, alpha = 1.0) -> None:
                super().__init__()
                self.alpha = alpha

            def forward(self, x):
                return custom_ops_lib.elu_custom(x, self.alpha)
        ```
        """
        inputs = self.signature.get("inputs", [])
        init_params = self.signature.get("init_params", [])

        # Generate forward parameters
        forward_params = ", ".join([inp["name"] for inp in inputs])

        # Generate __init__ signature and body
        if init_params:
            init_param_strs = []
            init_body_lines = []
            for param in init_params:
                # Build parameter string with optional default
                param_str = param["name"]
                if "default" in param and param["default"] is not None:
                    default_val = param["default"]
                    if isinstance(default_val, str):
                        param_str += f' = "{default_val}"'
                    else:
                        param_str += f" = {default_val}"
                init_param_strs.append(param_str)
                init_body_lines.append(f"        self.{param['name']} = {param['name']}")

            init_signature = ", ".join(init_param_strs)
            init_body = "\n".join(init_body_lines)
        else:
            init_signature = ""
            init_body = "        pass"

        # Generate custom op call args (all inputs + init_params as self.xxx)
        op_args = [inp["name"] for inp in inputs]
        for param in init_params:
            op_args.append(f"self.{param['name']}")
        op_args_str = ", ".join(op_args)

        # Generate model_src with local .so loading to avoid global pip conflicts
        # This ensures each project uses its own compiled custom_ops_lib
        # NOTE: project_path is hardcoded at generation time to work with exec()
        return f'''import sys
import os
import glob

# Priority load project-local custom_ops_lib to avoid global conflicts
# This enables parallel compilation without .so file conflicts
# Path hardcoded at generation time (exec() doesn't have __file__)
_project_path = "{project_path}"
_build_dirs = glob.glob(os.path.join(_project_path, "CppExtension", "build", "lib.*"))
if _build_dirs:
    sys.path.insert(0, _build_dirs[0])

import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, {init_signature}) -> None:
        super().__init__()
{init_body}

    def forward(self, {forward_params}):
        return custom_ops_lib.{self.op_custom}({op_args_str})
'''
