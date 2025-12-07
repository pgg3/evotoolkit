# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Component 3: host_operator_src generator.

Generates the host-side operator implementation including:
- TilingFunc
- InferShape
- InferDataType
- OpDef class
"""

from typing import Any, Dict, List

from .base import TemplateBase


class HostOperatorGenerator(TemplateBase):
    """Generate host operator implementation for Ascend C operator."""

    def generate(self, block_dim: int, tiling_func_body: str) -> str:
        """
        Generate host operator implementation.

        Args:
            block_dim: Number of parallel cores.
            tiling_func_body: Body of the TilingFunc function.

        Returns:
            Complete host operator implementation file content.
        """
        inputs = self.signature.get("inputs", [])
        outputs = self.signature.get("outputs", [])
        init_params = self.signature.get("init_params", [])

        input_defs = ""
        attr_defs = ""

        # Process all inputs
        for inp in inputs:
            if inp.get("is_tensor", True):
                ge_dtype = self._dtype_to_ge_datatype(inp.get("dtype", "float"))
                input_defs += f'        this->Input("{inp["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'
            else:
                # Scalar as attr
                attr_defs += self._gen_attr_def(inp)

        # Process init_params
        for param in init_params:
            if param.get("is_tensor", False):
                ge_dtype = self._dtype_to_ge_datatype(param.get("dtype", "float"))
                input_defs += f'        this->Input("{param["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'
            else:
                # Scalar as attr
                attr_defs += self._gen_attr_def(param)

        output_defs = ""
        for out in outputs:
            if out.get("is_tensor", True):
                ge_dtype = self._dtype_to_ge_datatype(out.get("dtype", "float"))
                output_defs += f'        this->Output("{out["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'

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
{input_defs}{output_defs}{attr_defs}
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }}
}};

OP_ADD({self.op_custom_capital});
}}
'''

    def default_tiling_func_body(self) -> str:
        """
        Default TilingFunc body for simple element-wise operators.

        Returns:
            Default tiling function body code.
        """
        return """
    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalLength *= shape.GetDim(i);
    }

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(BLOCK_DIM);
"""

    def add_scalar_params_to_tiling_func(
        self, base_func_body: str, scalar_params: List[Dict[str, Any]]
    ) -> str:
        """
        Add scalar params retrieval from attrs to tiling func body.

        Args:
            base_func_body: Base tiling function body.
            scalar_params: List of scalar parameter info.

        Returns:
            Tiling function body with scalar param handling added.
        """
        if not scalar_params:
            return base_func_body

        # Generate attr retrieval code
        attr_code = "\n    // Get scalar attrs\n"
        attr_code += "    const gert::RuntimeAttrs *attrs = context->GetAttrs();\n"

        for i, param in enumerate(scalar_params):
            cpp_type = self._dtype_to_cpp_type(param.get("dtype", "float"))
            field_name = self._to_camel_case(param["name"])
            attr_code += f"    const {cpp_type} *{field_name}Ptr = attrs->GetAttrPointer<{cpp_type}>({i});\n"
            attr_code += f"    tiling.set_{field_name}(*{field_name}Ptr);\n"

        return base_func_body + attr_code

    def _gen_attr_def(self, param: Dict[str, Any]) -> str:
        """
        Generate CANN attr definition for scalar parameter.

        Args:
            param: Parameter info dict with name, dtype, and optional default.

        Returns:
            C++ code for attr definition.
        """
        name = param["name"]
        dtype = param.get("dtype", "float")
        dtype_lower = dtype.lower()

        # Determine if optional (has default value)
        has_default = "default" in param and param["default"] is not None
        attr_type = "OPTIONAL" if has_default else "REQUIRED"
        default_val = param.get("default")

        if dtype_lower in ("float", "float32", "float16"):
            if has_default:
                return f'        this->Attr("{name}").AttrType({attr_type}).Float({default_val});\n'
            return f'        this->Attr("{name}").AttrType({attr_type}).Float();\n'
        elif dtype_lower in ("int", "int32", "int64"):
            if has_default:
                return f'        this->Attr("{name}").AttrType({attr_type}).Int({default_val});\n'
            return f'        this->Attr("{name}").AttrType({attr_type}).Int();\n'
        elif dtype_lower == "bool":
            bool_val = "true" if default_val else "false"
            if has_default:
                return f'        this->Attr("{name}").AttrType({attr_type}).Bool({bool_val});\n'
            return f'        this->Attr("{name}").AttrType({attr_type}).Bool();\n'
        else:
            if has_default:
                return f'        this->Attr("{name}").AttrType({attr_type}).Float({default_val});\n'
            return f'        this->Attr("{name}").AttrType({attr_type}).Float();\n'
