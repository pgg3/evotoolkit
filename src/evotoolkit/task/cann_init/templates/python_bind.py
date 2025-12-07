# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Component 5: python_bind_src generator.

Generates Python binding code using pybind11 for torch_npu integration.
"""

from typing import Any, Dict

from .base import TemplateBase


class PythonBindGenerator(TemplateBase):
    """Generate Python binding code for Ascend C operator."""

    def generate(self) -> str:
        """
        Generate Python binding code.

        Supports both tensor and scalar parameters uniformly.

        Returns:
            Complete pybind11 binding file content.

        Example output:
        ```cpp
        #include <torch/library.h>
        #include <torch/csrc/autograd/custom_function.h>
        #include "pytorch_npu_helper.hpp"
        #include <torch/extension.h>

        at::Tensor add_custom_impl_npu(const at::Tensor& x, const at::Tensor& y) {
            at::Tensor result = at::empty_like(x);
            EXEC_NPU_CMD(aclnnAddCustom, x, y, result);
            return result;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("add_custom", &add_custom_impl_npu, "add operator");
        }
        ```
        """
        inputs = self.signature.get("inputs", [])
        init_params = self.signature.get("init_params", [])

        param_parts = []
        first_tensor = None

        # Process all inputs
        for inp in inputs:
            if inp.get("is_tensor", True):
                param_parts.append(f"const at::Tensor& {inp['name']}")
                if first_tensor is None:
                    first_tensor = inp['name']
            else:
                cpp_type = self._dtype_to_cpp_type(inp.get("dtype", "float"))
                param_parts.append(f"{cpp_type} {inp['name']}")

        # Process init_params
        for param in init_params:
            if param.get("is_tensor", False):
                param_parts.append(f"const at::Tensor& {param['name']}")
                if first_tensor is None:
                    first_tensor = param['name']
            else:
                cpp_type = self._dtype_to_cpp_type(param.get("dtype", "float"))
                param_parts.append(f"{cpp_type} {param['name']}")

        all_params = ", ".join(param_parts)

        # Generate args for EXEC_NPU_CMD (all params + result)
        all_args = [inp["name"] for inp in inputs] + [param["name"] for param in init_params]
        exec_args = ", ".join(all_args + ["result"])

        # Use first tensor for result allocation
        if first_tensor is None:
            first_tensor = "x"

        return f'''#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor {self.op_custom}_impl_npu({all_params}) {{
    at::Tensor result = at::empty_like({first_tensor});
    EXEC_NPU_CMD(aclnn{self.op_custom_capital}, {exec_args});
    return result;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{self.op_custom}", &{self.op_custom}_impl_npu, "{self.op_name} operator");
}}
'''
