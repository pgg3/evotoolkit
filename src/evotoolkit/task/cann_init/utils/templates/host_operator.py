# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from typing import Any, Dict, List, Optional

from .base import TemplateBase


class HostOperatorGenerator(TemplateBase):

    def generate(
        self,
        tiling_func_body: str,
        infer_shape_body: str,
        soc_versions: Optional[List[str]] = None,
        tiling_func_includes: Optional[List[str]] = None,
    ) -> str:
        inputs = self.signature.get("inputs", [])
        outputs = self.signature.get("outputs", [])
        init_params = self.signature.get("init_params", [])

        input_defs = ""
        attr_defs = ""

        for inp in inputs:
            if inp.get("is_tensor", True):
                ge_dtype = self._dtype_to_ge_datatype(inp.get("dtype", "float"))
                input_defs += f'        this->Input("{inp["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'
            else:
                attr_defs += self._gen_attr_def(inp)

        for param in init_params:
            if param.get("is_tensor", False):
                ge_dtype = self._dtype_to_ge_datatype(param.get("dtype", "float"))
                input_defs += f'        this->Input("{param["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'
            else:
                attr_defs += self._gen_attr_def(param)

        output_defs = ""
        for out in outputs:
            if out.get("is_tensor", True):
                ge_dtype = self._dtype_to_ge_datatype(out.get("dtype", "float"))
                output_defs += f'        this->Output("{out["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'

        if soc_versions is None:
            soc_versions = ["ascend910b"]
        aicore_configs = "\n".join(
            f'        this->AICore().AddConfig("{soc}");' for soc in soc_versions
        )

        # Generate extra includes for TilingFunc
        extra_includes = ""
        if tiling_func_includes:
            for inc in tiling_func_includes:
                extra_includes += f'#include "{inc}"\n'

        return f'''#include "{self.op_custom}_tiling.h"
#include "register/op_def_registry.h"
{extra_includes}

namespace optiling {{

/**
 * TilingFunc - Compute tiling parameters for the operator.
 *
 * Steps:
 *   1. Get input shape from context
 *   2. Compute tiling parameters (totalLength, tileNum, etc.)
 *   3. Set blockDim via context->SetBlockDim()
 *   4. Serialize TilingData to buffer
 *   5. (Optional) Set workspace size
 */
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{{
{tiling_func_body}
}}

}}

namespace ge {{

/**
 * InferShape - Infer output shape from input shapes.
 */
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{{
{infer_shape_body}
}}

}}

namespace ops {{

class {self.op_custom_capital} : public OpDef {{
public:
    explicit {self.op_custom_capital}(const char* name) : OpDef(name)
    {{
{input_defs}{output_defs}{attr_defs}
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
{aicore_configs}
    }}
}};

OP_ADD({self.op_custom_capital});

}}
'''

    def _gen_attr_def(self, param: Dict[str, Any]) -> str:
        name = param["name"]
        dtype = param.get("dtype", "float")
        dtype_lower = dtype.lower()

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
