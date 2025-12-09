# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Code Implementation Prompts for Joint Branch

This module contains prompts for the code generation phase:
- Kernel implementation prompt (generates kernel C++ code)
- Tiling implementation prompt (generates host-side tiling code)
"""


class ImplPromptsMixin:
    """Prompts for code implementation phase"""

    def get_kernel_impl_prompt(self, plan: dict, knowledge: dict, python_ref: str) -> str:
        """Generate prompt for Kernel implementation"""
        return f"""
You are an Ascend C Kernel Development Expert. Please generate the complete kernel code.

## Python Reference
```python
{python_ref}
```

## Joint Plan
{plan}

## Available Knowledge (API Documentation / Example Code)
{knowledge}

## Code Requirements
1. Use Ascend C APIs
2. Implement the standard structure:
   - class KernelXxx: Init, Process, CopyIn, Compute, CopyOut
   - extern "C" __global__ entry function
3. Handle tiling parameters correctly
4. Use pipeline optimization

## Code Template
```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelXxx {{
public:
    __aicore__ inline KernelXxx() {{}}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {{
        // Initialize GlobalTensor, calculate tiling parameters
    }}
    __aicore__ inline void Process() {{
        // Main loop: CopyIn -> Compute -> CopyOut
    }}
private:
    __aicore__ inline void CopyIn(int32_t progress) {{ /* DataCopy GM -> UB */ }}
    __aicore__ inline void Compute(int32_t progress) {{ /* Compute */ }}
    __aicore__ inline void CopyOut(int32_t progress) {{ /* DataCopy UB -> GM */ }}

    // Member variables
    GlobalTensor<half> xGm, yGm, zGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, 2> outQueue;
}};

extern "C" __global__ __aicore__ void xxx_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tilingData, tiling);
    KernelXxx op;
    op.Init(x, y, z, tilingData.totalLength);
    op.Process();
}}
```

Please return the complete kernel C++ code.
"""

    def get_tiling_impl_prompt(self, plan: dict, knowledge: dict) -> str:
        """Generate prompt for Tiling implementation"""
        return f"""
You are an Ascend C Host-side Development Expert. Please generate tiling-related code.

## Joint Plan
{plan}

## Available Knowledge
{knowledge}

## Two Files to Generate

### 1. tiling.h (Tiling Data Structure)
```cpp
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF(XxxTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Xxx, XxxTilingData)
}}
```

### 2. op_host.cpp (Tiling Calculation + InferShape)
```cpp
#include "xxx_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {{
static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
    // Calculate tiling parameters
    XxxTilingData tiling;
    tiling.set_totalLength(...);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), ...);
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ops {{
class Xxx : public OpDef {{
public:
    Xxx(const char* name) : OpDef(name) {{
        // Define inputs and outputs
        this->Input("x").ParamType(REQUIRED).DataType({{ge::DT_FLOAT16}});
        this->Output("z").ParamType(REQUIRED).DataType({{ge::DT_FLOAT16}});
        // Register tiling
        this->SetInferShape(ge::InferShape).SetTiling(optiling::TilingFunc);
    }}
}};
OP_ADD(Xxx);
}}
```

## Return JSON Format
```json
{{
  "host_tiling_src": "Complete tiling.h code",
  "host_operator_src": "Complete op_host.cpp code"
}}
```
"""
