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

import json
from typing import Any, Dict, List, Optional


class AscendCTemplateGenerator:
    """
    Generate Ascend C operator code from templates.

    Given an operator signature and kernel code, generates all 6 components
    needed for a complete Ascend C operator.
    """

    def __init__(self, signature: Dict[str, Any]):
        """
        Initialize with operator signature.

        Args:
            signature: Operator signature containing:
                - op_name: Operator name (e.g., "add")
                - inputs: List of input tensor info
                - outputs: List of output tensor info
                - dtypes: Supported data types
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

    def generate(
        self,
        kernel_src: str,
        block_dim: int = 8,
        tiling_fields: Optional[List[Dict[str, str]]] = None,
        tiling_func_body: Optional[str] = None,
        host_tiling_src: Optional[str] = None,
        host_operator_src: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate all 6 code components.

        Tiling Modes:
        1. Direct mode: If host_tiling_src or host_operator_src provided, use directly
        2. Template mode: Use tiling_fields and tiling_func_body to generate
        3. Default mode: Use built-in defaults (for element-wise operators)

        Args:
            kernel_src: Kernel code (from LLM)
            block_dim: Number of parallel cores
            tiling_fields: Custom tiling fields (template mode)
            tiling_func_body: Custom TilingFunc body (template mode)
            host_tiling_src: Complete tiling header (direct mode)
            host_operator_src: Complete host operator (direct mode)

        Returns:
            Dictionary with all 6 code components
        """
        # Generate host_tiling_src
        if host_tiling_src is None:
            # Template mode or default mode
            fields = tiling_fields if tiling_fields is not None else self._default_tiling_fields()
            host_tiling_src = self._gen_host_tiling(fields)

        # Generate host_operator_src
        if host_operator_src is None:
            # Template mode or default mode
            func_body = tiling_func_body if tiling_func_body is not None else self._default_tiling_func_body()
            host_operator_src = self._gen_host_operator(block_dim, func_body)

        return {
            "project_json_src": self._gen_project_json(),
            "host_tiling_src": host_tiling_src,
            "host_operator_src": host_operator_src,
            "kernel_src": kernel_src,
            "python_bind_src": self._gen_python_bind(),
            "model_src": self._gen_model_src(),
        }

    def _default_tiling_fields(self) -> List[Dict[str, str]]:
        """Default tiling fields for simple element-wise operators."""
        return [
            {"name": "totalLength", "type": "uint32_t"},
            {"name": "tileNum", "type": "uint32_t"},
        ]

    def _default_tiling_func_body(self) -> str:
        """Default TilingFunc body for simple element-wise operators."""
        return """
    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalLength *= shape.GetDim(i);
    }

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(BLOCK_DIM);
"""

    def _gen_project_json(self) -> str:
        """Generate project JSON configuration."""
        inputs = self.signature.get("inputs", [])
        outputs = self.signature.get("outputs", [])

        # Note: CANN uses "float" instead of "float32"
        # Use single dtype to avoid opbuild issues
        input_desc = []
        for inp in inputs:
            input_desc.append({
                "name": inp["name"],
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"],
            })

        output_desc = []
        for out in outputs:
            output_desc.append({
                "name": out["name"],
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"],
            })

        config = [{
            "op": self.op_custom_capital,
            "language": "cpp",
            "input_desc": input_desc,
            "output_desc": output_desc,
        }]

        return json.dumps(config, indent=4)

    def _gen_host_tiling(self, tiling_fields: List[Dict[str, str]]) -> str:
        """Generate host tiling header."""
        fields_code = ""
        for field in tiling_fields:
            fields_code += f"    TILING_DATA_FIELD_DEF({field['type']}, {field['name']});\n"

        return f'''#ifndef {self.op_custom.upper()}_TILING_H
#define {self.op_custom.upper()}_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF({self.op_custom_capital}TilingData)
{fields_code}END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({self.op_custom_capital}, {self.op_custom_capital}TilingData)
}}

#endif // {self.op_custom.upper()}_TILING_H
'''

    def _gen_host_operator(self, block_dim: int, tiling_func_body: str) -> str:
        """Generate host operator implementation."""
        inputs = self.signature.get("inputs", [])
        outputs = self.signature.get("outputs", [])

        # Generate input/output definitions (use single dtype to match project.json)
        input_defs = ""
        for inp in inputs:
            input_defs += f'        this->Input("{inp["name"]}").ParamType(REQUIRED).DataType({{ge::DT_FLOAT}}).Format({{ge::FORMAT_ND}});\n'

        output_defs = ""
        for out in outputs:
            output_defs += f'        this->Output("{out["name"]}").ParamType(REQUIRED).DataType({{ge::DT_FLOAT}}).Format({{ge::FORMAT_ND}});\n'

        return f'''#include "{self.op_custom}_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {{
const uint32_t BLOCK_DIM = {block_dim};

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{{
    {self.op_custom_capital}TilingData tiling;
{tiling_func_body}
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(BLOCK_DIM);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ge {{
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{{
    const ge::DataType x1_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1_dtype);
    return GRAPH_SUCCESS;
}}
}}

namespace ops {{
class {self.op_custom_capital} : public OpDef {{
public:
    explicit {self.op_custom_capital}(const char* name) : OpDef(name)
    {{
{input_defs}{output_defs}
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }}
}};

OP_ADD({self.op_custom_capital});
}}
'''

    def _gen_python_bind(self) -> str:
        """
        Generate Python binding code.

        Supports both tensor inputs and scalar parameters (from __init__).
        """
        inputs = self.signature.get("inputs", [])
        init_params = self.signature.get("init_params", [])

        # Generate tensor input parameters
        param_parts = [f"const at::Tensor& {inp['name']}" for inp in inputs]

        # Add scalar parameters from __init__
        for param in init_params:
            cpp_type = self._dtype_to_cpp_type(param.get("dtype", "float"))
            param_parts.append(f"{cpp_type} {param['name']}")

        all_params = ", ".join(param_parts)

        # Generate args for EXEC_NPU_CMD (tensors + scalars + result)
        input_args = ", ".join([inp["name"] for inp in inputs])
        scalar_args = ", ".join([param["name"] for param in init_params])

        if scalar_args:
            exec_args = f"{input_args}, {scalar_args}, result"
        else:
            exec_args = f"{input_args}, result"

        # Determine first input for result allocation
        first_input = inputs[0]["name"] if inputs else "x"

        return f'''#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor {self.op_custom}_impl_npu({all_params}) {{
    at::Tensor result = at::empty_like({first_input});
    EXEC_NPU_CMD(aclnn{self.op_custom_capital}, {exec_args});
    return result;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{self.op_custom}", &{self.op_custom}_impl_npu, "{self.op_name} operator");
}}
'''

    def _dtype_to_cpp_type(self, dtype: str) -> str:
        """Convert Python dtype string to C++ type."""
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

    def _gen_model_src(self) -> str:
        """
        Generate test model code (ModelNew class).

        ModelNew must have the same interface as Model:
        - Same __init__ parameters
        - Same forward parameters
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

        return f'''import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, {init_signature}) -> None:
        super().__init__()
{init_body}

    def forward(self, {forward_params}):
        return custom_ops_lib.{self.op_custom}({op_args_str})
'''
