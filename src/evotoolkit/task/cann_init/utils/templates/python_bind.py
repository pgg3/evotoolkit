# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from .base import TemplateBase


class PythonBindGenerator(TemplateBase):
    def generate(self, output_alloc_code: str) -> str:
        """Generate Python binding code.

        Args:
            output_alloc_code: Required. Code for allocating output tensor(s).
                              Must define 'result' variable.
        """
        inputs = self.signature.get("inputs", [])
        init_params = self.signature.get("init_params", [])

        param_parts = []

        for inp in inputs:
            if inp.get("is_tensor", True):
                param_parts.append(f"const at::Tensor& {inp['name']}")
            else:
                cpp_type = self._dtype_to_cpp_type(inp.get("dtype", "float"))
                param_parts.append(f"{cpp_type} {inp['name']}")

        for param in init_params:
            if param.get("is_tensor", False):
                param_parts.append(f"const at::Tensor& {param['name']}")
            else:
                cpp_type = self._dtype_to_cpp_type(param.get("dtype", "float"))
                param_parts.append(f"{cpp_type} {param['name']}")

        all_params = ", ".join(param_parts)

        all_args = [inp["name"] for inp in inputs] + [param["name"] for param in init_params]
        exec_args = ", ".join(all_args + ["result"])

        return f'''#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor {self.op_custom}_impl_npu({all_params}) {{
    {output_alloc_code}
    EXEC_NPU_CMD(aclnn{self.op_custom_capital}, {exec_args});
    return result;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{self.op_custom}", &{self.op_custom}_impl_npu, "{self.op_name} operator");
}}
'''
