# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import re
from typing import Dict, List, Union

from .base import TemplateBase


class HostTilingGenerator(TemplateBase):
    def generate(self, tiling_fields: List[Dict[str, Union[str, int]]]) -> str:
        fields_code = ""
        for field in tiling_fields:
            fields_code += f"    {self._field_to_macro(field)}\n"

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

    def _field_to_macro(self, field: Dict[str, Union[str, int]]) -> str:
        name = field["name"]
        type_str = field["type"]
        size = field.get("size")
        is_struct = field.get("is_struct", False)

        # Struct field: TILING_DATA_FIELD_DEF_STRUCT(StructType, name)
        if is_struct:
            return f"TILING_DATA_FIELD_DEF_STRUCT({type_str}, {name});"

        # Array with explicit size: {"name": "x", "type": "int64_t", "size": 4}
        if size is not None:
            return f"TILING_DATA_FIELD_DEF_ARR({type_str}, {size}, {name});"

        # Array notation: {"name": "x", "type": "int64_t[4]"}
        arr_match = re.match(r"(\w+)\[(\d+)\]", type_str)
        if arr_match:
            base_type = arr_match.group(1)
            arr_size = arr_match.group(2)
            return f"TILING_DATA_FIELD_DEF_ARR({base_type}, {arr_size}, {name});"

        # Scalar field
        return f"TILING_DATA_FIELD_DEF({type_str}, {name});"
