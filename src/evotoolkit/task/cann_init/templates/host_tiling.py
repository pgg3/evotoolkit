# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Component 2: host_tiling_src generator.

Generates the tiling data structure definition header file.
"""

from typing import Any, Dict, List

from .base import TemplateBase


class HostTilingGenerator(TemplateBase):
    """Generate host tiling header for Ascend C operator."""

    def generate(self, tiling_fields: List[Dict[str, str]]) -> str:
        """
        Generate host tiling header.

        Args:
            tiling_fields: List of tiling field definitions.
                Each field is a dict with 'name' and 'type' keys.
                Example: [{"name": "totalLength", "type": "uint32_t"}]

        Returns:
            Complete tiling header file content.

        Example output:
        ```cpp
        #ifndef ADD_CUSTOM_TILING_H
        #define ADD_CUSTOM_TILING_H

        #include "register/tilingdata_base.h"

        namespace optiling {
        BEGIN_TILING_DATA_DEF(AddCustomTilingData)
            TILING_DATA_FIELD_DEF(uint32_t, totalLength);
            TILING_DATA_FIELD_DEF(uint32_t, tileNum);
        END_TILING_DATA_DEF;

        REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
        }

        #endif // ADD_CUSTOM_TILING_H
        ```
        """
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

    def default_tiling_fields(self) -> List[Dict[str, str]]:
        """
        Default tiling fields for simple element-wise operators.

        Returns:
            List of default tiling field definitions.
        """
        return [
            {"name": "totalLength", "type": "uint32_t"},
            {"name": "tileNum", "type": "uint32_t"},
        ]

    def scalar_params_to_tiling_fields(self, scalar_params: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Convert scalar params to tiling field definitions.

        Args:
            scalar_params: List of scalar parameter info from signature.

        Returns:
            List of tiling field definitions for scalar params.
        """
        fields = []
        for param in scalar_params:
            cpp_type = self._dtype_to_cpp_type(param.get("dtype", "float"))
            # Convert to camelCase for tiling field
            field_name = self._to_camel_case(param["name"])
            fields.append({"name": field_name, "type": cpp_type})
        return fields
