# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Code Implementation Prompts for Joint Branch

This module contains prompts for the three-stage code generation:
1. Tiling Header (tiling.h) - Tiling data structure definition (NO knowledge needed)
2. Tiling Host (op_host.cpp) - Tiling calculation + InferShape
3. Kernel (op_kernel.cpp) - Kernel implementation

Design Pattern (following pybind.py):
- Each stage has an `assemble_*` method that takes variable parts and produces complete code
- Prompts ask LLM to generate ONLY the variable parts
- Fixed structure is controlled programmatically, preventing LLM mistakes
"""

from typing import List, Optional

from evotoolkit.task.cann_init.method_interface.prompts.phase0 import (
    _format_signature_for_kernel,
)


def _to_pascal_case(name: str) -> str:
    """Convert operator name to PascalCase.

    CANN convention:
    - "SDPA" -> "Sdpa"
    - "add" -> "Add"
    - "layer_norm" -> "LayerNorm"
    """
    # Handle underscore-separated names
    if "_" in name:
        return "".join(word.capitalize() for word in name.split("_"))
    # Handle all-caps (SDPA -> Sdpa)
    if name.isupper():
        return name.capitalize()
    # Handle already PascalCase or mixed
    return name[0].upper() + name[1:] if name else name


def _format_tiling_fields(tiling_fields: List[dict]) -> str:
    """Format tiling fields for prompt display.

    Args:
        tiling_fields: List of {"name": str, "type": str, "purpose": str}

    Returns:
        Formatted string like:
        - totalLength: uint32_t // total number of elements
        - tileNum: uint32_t // number of tiles
    """
    if not tiling_fields:
        return "(no custom fields specified)"

    lines = []
    for field in tiling_fields:
        if isinstance(field, dict):
            name = field.get("name", "unknown")
            ftype = field.get("type", "uint32_t")
            purpose = field.get("purpose", "")
            lines.append(f"- {name}: {ftype} // {purpose}")
        elif isinstance(field, str):
            lines.append(f"- {field}")
    return "\n".join(lines) if lines else "(none)"


def _format_joint_plan(plan: dict) -> dict:
    """Extract and format joint plan components."""
    return {
        "tiling_strategy": plan.get("tiling_strategy", "default"),
        "tiling_fields": _format_tiling_fields(plan.get("tiling_fields", [])),
        "tiling_fields_raw": plan.get("tiling_fields", []),
        "tiling_execution": plan.get("tiling_execution", "(not specified)"),
        "kernel_pseudocode": plan.get("kernel_pseudocode", "(not specified)"),
        "kernel_design": plan.get("kernel_design", "(not specified)"),
        "tiling_proposal": plan.get("tiling_proposal", "(not specified)"),
    }


class ImplPromptsMixin:
    """Prompts for three-stage code implementation.

    Design Pattern (following pybind.py):
    - `assemble_*` methods: Take variable parts, produce complete code
    - `get_*_prompt` methods: Ask LLM for only the variable parts

    Stage 1: tiling.h
        - assemble_tiling_header() + get_tiling_header_prompt()
        - Variable: field definitions only

    Stage 2: op_host.cpp
        - assemble_tiling_host() + get_tiling_host_prompt()
        - Variable: TilingFunc body, Input/Output definitions

    Stage 3: op_kernel.cpp
        - assemble_kernel_impl() + get_kernel_impl_prompt()
        - Variable: Init/Process bodies, private methods (flexible), member variables
    """

    # =========================================================================
    # Stage 1: Tiling Header (tiling.h)
    # =========================================================================

    def assemble_tiling_header(
        self,
        op_name: str,
        field_definitions: str,
    ) -> str:
        """Assemble complete tiling.h from field definitions.

        Args:
            op_name: Operator name (e.g., "SDPA")
            field_definitions: Field definition lines from LLM
                Example: "    TILING_DATA_FIELD_DEF(uint32_t, batchSize);\n    ..."

        Returns:
            Complete tiling.h source code

        CANN Naming Convention:
            - op_name="SDPA" -> class: SdpaCustomTilingData, register: SdpaCustom
            - op_name="add" -> class: AddCustomTilingData, register: AddCustom
        """
        # Convert to CANN naming convention: SDPA -> Sdpa, add -> Add
        op_pascal = _to_pascal_case(op_name)
        op_lower = op_name.lower()
        tiling_class = f"{op_pascal}CustomTilingData"
        register_name = f"{op_pascal}Custom"
        header_guard = f"{op_lower.upper()}_CUSTOM_TILING_H"

        return f"""#ifndef {header_guard}
#define {header_guard}
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF({tiling_class})
{field_definitions}
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({register_name}, {tiling_class})
}}

#endif // {header_guard}
"""

    def get_tiling_header_prompt(
        self,
        plan: dict,
        context: dict,
    ) -> str:
        """Generate prompt for tiling.h field definitions.

        NOTE: This stage does NOT need knowledge - it's pure structure definition.

        Args:
            plan: Joint plan dict from _extract_joint_plan()
            context: Contains 'signature', 'op_name', etc.

        Returns:
            Prompt string for generating tiling.h field definitions
        """
        formatted = _format_joint_plan(plan)
        op_name = context.get("op_name", "CustomOp")
        formatted_sig = _format_signature_for_kernel(context.get("signature"))

        # Show the template with placeholder
        template_code = self.assemble_tiling_header(op_name, "    // === YOUR FIELD DEFINITIONS HERE ===")

        return f"""## Role
You are an Ascend C expert generating the **tiling.h** header file.

## Task
Define the tiling data structure fields based on the joint plan.

## Input

### Operator
{formatted_sig}

### Tiling Fields (from Joint Plan)
{formatted["tiling_fields"]}

### Tiling Execution (reference)
```
{formatted["tiling_execution"]}
```

## Fixed Code (you cannot modify)

```cpp
{template_code}```

## Your Task

Output ONLY the field definitions that replace `// === YOUR FIELD DEFINITIONS HERE ===`.

### Field Format
Each field must use: `TILING_DATA_FIELD_DEF(type, name);`

Supported types: `uint32_t`, `int32_t`, `float`, `uint64_t`

## Response Format

<response>
    TILING_DATA_FIELD_DEF(uint32_t, fieldName1);
    TILING_DATA_FIELD_DEF(uint32_t, fieldName2);
    ...
</response>

## Example

For Softmax with fields `batchSize`, `featureDim`, `rowsPerCore`:

<response>
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, featureDim);
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
</response>

Now output the field definitions for `{op_name}`. Output ONLY the `<response>` block:
"""

    # =========================================================================
    # Stage 2: Tiling Host (op_host.cpp)
    # =========================================================================

    def assemble_tiling_host(
        self,
        op_name: str,
        tiling_func_body: str,
        input_output_defs: str,
        infer_shape_body: str = "",
    ) -> str:
        """Assemble complete op_host.cpp from variable parts.

        Args:
            op_name: Operator name (e.g., "SDPA")
            tiling_func_body: TilingFunc body from LLM (calculation logic)
            input_output_defs: Input/Output definitions from LLM
            infer_shape_body: InferShape body from LLM (optional, defaults to output=input)

        Returns:
            Complete op_host.cpp source code

        CANN Naming Convention:
            - op_name="SDPA" -> op_lower="sdpa", op_class="SdpaCustom"
        """
        op_lower = op_name.lower()
        op_pascal = _to_pascal_case(op_name)
        op_class = f"{op_pascal}Custom"

        # Default InferShape: output shape = input shape
        if not infer_shape_body or not infer_shape_body.strip():
            infer_shape_body = """    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;"""

        return f"""#include "{op_lower}_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {{

static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
{tiling_func_body}
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ge {{
static ge::graphStatus InferShape(gert::InferShapeContext* context) {{
{infer_shape_body}
    return GRAPH_SUCCESS;
}}
}}

namespace ops {{
class {op_class} : public OpDef {{
public:
    explicit {op_class}(const char* name) : OpDef(name) {{
{input_output_defs}
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }}
}};
OP_ADD({op_class});
}}
"""

    def get_tiling_host_prompt(
        self,
        plan: dict,
        context: dict,
        knowledge: str,
        tiling_header: str,
    ) -> str:
        """Generate prompt for op_host.cpp variable parts.

        Context Design (IMPORTANT):
        - DO NOT include python_ref: tiling_execution already contains the
          translated logic. Including python_ref would be redundant and may
          confuse the LLM with inconsistent representations.
        - DO NOT include tiling_proposal: it's the discussion process,
          tiling_execution is the final agreed result.

        Required context:
        - tiling_header: field names from Stage 1
        - tiling_execution: how to calculate tiling parameters
        - signature: for Input/Output definitions
        - knowledge: API syntax reference

        Args:
            plan: Joint plan dict
            context: Contains 'signature', 'op_name', etc.
            knowledge: Retrieved knowledge context
            tiling_header: Generated tiling.h content from Stage 1

        Returns:
            Prompt string for generating op_host.cpp variable parts
        """
        formatted = _format_joint_plan(plan)
        op_name = context.get("op_name", "CustomOp")
        formatted_sig = _format_signature_for_kernel(context.get("signature"))
        # NOTE: python_ref is intentionally NOT used here.
        # tiling_execution already contains the translated calculation logic.

        # Compute correct tiling class name (CANN convention)
        op_pascal = _to_pascal_case(op_name)
        tiling_class = f"{op_pascal}CustomTilingData"

        # Show the template with placeholders
        template_code = self.assemble_tiling_host(
            op_name,
            "    // === TILING_FUNC_BODY ===",
            "        // === INPUT_OUTPUT_DEFS ===",
            ""  # InferShape is auto-translated from pybind
        )

        # Shape inference note (InferShape is now handled by InferShapeTranslator)
        shape_section = """### Shape Inference
- InferShape is auto-generated from pybind branch output
- You do NOT need to provide shape inference code"""

        return f"""## Role
You are an Ascend C host-side development expert generating the **op_host.cpp** file.

## Task
Implement tiling calculation logic and operator input/output definitions.

## Input

### Operator
{formatted_sig}

{shape_section}

### Tiling Execution (from Joint Plan)
This describes how to calculate each tiling field:
```
{formatted["tiling_execution"]}
```

### Generated tiling.h (Stage 1 output)
```cpp
{tiling_header}
```

### Available Knowledge
{knowledge}

## Fixed Code (you cannot modify structure)

```cpp
{template_code}```

## Your Task

Provide the following 2 parts:

### Part 1: TILING_FUNC_BODY (required)
The body of TilingFunc (excluding return statement).
Must include:
- Shape extraction: `context->GetInputShape(i)->GetStorageShape()`
- Tiling calculation logic
- Create and populate tiling struct: `{tiling_class} tiling;`
- Set fields: `tiling.set_fieldName(value);`
- Set block dim: `context->SetBlockDim(n);`
- Save tiling (fixed pattern below)

Save Tiling Pattern (always use this):
```cpp
tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
```

### Part 2: INPUT_OUTPUT_DEFS (required)
Input and output definitions for the OpDef class.
Pattern:
- Input: `this->Input("name").ParamType(REQUIRED).DataType({{ge::DT_FLOAT}});`
- Output: `this->Output("name").ParamType(REQUIRED).DataType({{ge::DT_FLOAT}});`

Common DataTypes: `ge::DT_FLOAT16`, `ge::DT_FLOAT`, `ge::DT_INT32`, `ge::DT_INT64`, `ge::DT_BOOL`

## Response Format

<tiling_func_body>
    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t dim0 = shape.GetDim(0);
    ...
    {tiling_class} tiling;
    tiling.set_field1(value1);
    ...
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
</tiling_func_body>

<input_output_defs>
        this->Input("x").ParamType(REQUIRED).DataType({{ge::DT_FLOAT}});
        this->Output("y").ParamType(REQUIRED).DataType({{ge::DT_FLOAT}});
</input_output_defs>

## Example (Softmax)

<tiling_func_body>
    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t batchSize = shape.GetDim(0);
    uint32_t featureDim = shape.GetDim(1);
    uint32_t coreNum = 8;
    uint32_t rowsPerCore = (batchSize + coreNum - 1) / coreNum;

    SoftmaxCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_featureDim(featureDim);
    tiling.set_rowsPerCore(rowsPerCore);

    context->SetBlockDim(std::min(coreNum, batchSize));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
</tiling_func_body>

<input_output_defs>
        this->Input("x").ParamType(REQUIRED).DataType({{ge::DT_FLOAT16, ge::DT_FLOAT}});
        this->Output("y").ParamType(REQUIRED).DataType({{ge::DT_FLOAT16, ge::DT_FLOAT}});
</input_output_defs>

Now output the 2 parts for `{op_name}`:
"""

    # =========================================================================
    # Stage 3: Kernel Implementation (op_kernel.cpp)
    # =========================================================================

    def assemble_kernel_impl(
        self,
        op_name: str,
        init_params: str,
        init_body: str,
        process_body: str,
        private_methods: str,
        member_vars: str,
        global_func_params: str,
        init_call_args: str,
    ) -> str:
        """Assemble complete op_kernel.cpp from variable parts.

        Args:
            op_name: Operator name (e.g., "SDPA")
            init_params: Init function parameters
            init_body: Init function body
            process_body: Process function body (main loop logic)
            private_methods: Private helper methods (flexible - can be any methods)
            member_vars: Member variable declarations
            global_func_params: Parameters for the global function (before workspace, tiling)
            init_call_args: Arguments passed to op.Init()

        Returns:
            Complete op_kernel.cpp source code
        """
        op_lower = op_name.lower()

        return f"""#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class Kernel{op_name} {{
public:
    __aicore__ inline Kernel{op_name}() {{}}
    __aicore__ inline void Init({init_params}) {{
{init_body}
    }}
    __aicore__ inline void Process() {{
{process_body}
    }}

private:
{private_methods}

private:
{member_vars}
}};

extern "C" __global__ __aicore__ void {op_lower}_custom(
    {global_func_params}GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tilingData, tiling);
    Kernel{op_name} op;
    op.Init({init_call_args});
    op.Process();
}}
"""

    def get_kernel_impl_prompt(
        self,
        plan: dict,
        context: dict,
        knowledge: str,
        tiling_header: Optional[str] = None,
    ) -> str:
        """Generate prompt for op_kernel.cpp variable parts.

        Context Design (IMPORTANT):
        - DO NOT include python_ref: kernel_pseudocode already contains the
          translated computation logic. Including python_ref would be redundant
          and may confuse the LLM with inconsistent representations.
        - DO NOT include kernel_design: it's the discussion process,
          kernel_pseudocode is the final agreed implementation plan.

        Required context:
        - kernel_pseudocode: the computation logic to implement
        - tiling_execution: loop structure and data flow
        - tiling_header: available tiling parameters
        - knowledge: API syntax reference (IMPORTANT for correct API usage)

        Args:
            plan: Joint plan dict
            context: Contains 'signature', 'op_name', etc.
            knowledge: Retrieved knowledge context
            tiling_header: Generated tiling.h content (None for default tiling)

        Returns:
            Prompt string for generating op_kernel.cpp variable parts
        """
        formatted = _format_joint_plan(plan)
        op_name = context.get("op_name", "CustomOp")
        formatted_sig = _format_signature_for_kernel(context.get("signature"))
        # NOTE: python_ref is intentionally NOT used here.
        # kernel_pseudocode already contains the translated computation logic.

        # Tiling section - unified format for both default and custom
        if tiling_header:
            # Custom tiling: show generated header + custom execution logic
            tiling_section = f"""### Tiling (Custom Generated)

**Tiling Header (tiling.h)**:
```cpp
{tiling_header}
```

**Host Tiling Execution** (how fields are calculated):
```
{formatted["tiling_execution"] or "(see header fields above)"}
```

**Kernel Usage**: See Kernel Pseudocode section below for data access pattern."""
        else:
            # Default tiling: show the standard fields and host-side calculation
            tiling_section = """### Tiling (Default Template)

**Tiling Fields** (via `GET_TILING_DATA(tilingData, tiling)`):
- `tilingData.totalLength`: total elements (all dimensions flattened)
- `tilingData.tileNum`: BLOCK_DIM (number of cores, e.g., 8)

**Host-Side Calculation** (default template does this):
```cpp
auto shape = context->GetInputShape(0)->GetStorageShape();
uint32_t totalLength = 1;
for (size_t i = 0; i < shape.GetDimNum(); i++) {
    totalLength *= shape.GetDim(i);
}
tiling.set_totalLength(totalLength);
tiling.set_tileNum(BLOCK_DIM);  // BLOCK_DIM = 8
```

**Kernel-Side Usage**:
```cpp
// Calculate per-core work
uint32_t blockLength = totalLength / GetBlockNum();
// Calculate per-tile size
uint32_t tileLength = blockLength / tileNum / BUFFER_NUM;
// Loop count
int32_t loopCount = tileNum * BUFFER_NUM;
```"""

        # Show the template with placeholders
        template_code = self.assemble_kernel_impl(
            op_name,
            "/* INIT_PARAMS */",
            "        // INIT_BODY",
            "        // PROCESS_BODY",
            "    // PRIVATE_METHODS",
            "    // MEMBER_VARS",
            "/* GLOBAL_FUNC_PARAMS */ ",
            "/* INIT_CALL_ARGS */"
        )

        return f"""## Role
You are an Ascend C kernel development expert generating the **op_kernel.cpp** file.

## Task
Implement the kernel based on the pseudocode and available knowledge.

## Input

### Operator
{formatted_sig}

{tiling_section}

### Kernel Pseudocode (from Joint Plan)
```
{formatted["kernel_pseudocode"]}
```

### Available Knowledge (API & Example Reference)
{knowledge}

## API Quick Reference

| Pattern | Code |
|---------|------|
| GetBlockIdx | `GetBlockIdx()` - current core index (0 to BlockNum-1) |
| GetBlockNum | `GetBlockNum()` - total number of cores |
| GlobalTensor | `xGm.SetGlobalBuffer((__gm__ T*)addr + offset, length)` |
| Pipe init | `pipe.InitBuffer(queue, BUFFER_NUM, size)` |
| Alloc/Free | `queue.AllocTensor<T>()` / `queue.FreeTensor(tensor)` |
| DataCopy | `DataCopy(dst, src[offset], count)` |
| EnQue/DeQue | `queue.EnQue(tensor)` / `queue.DeQue<T>()` |
| Compute | `Op(dst, src, count)` - e.g., `Relu`, `Add`, `Mul` |

| Type | Declaration |
|------|-------------|
| Pipe | `TPipe pipe;` |
| Input Queue | `TQue<TPosition::VECIN, BUFFER_NUM> inQueue;` |
| Output Queue | `TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;` |
| GlobalTensor | `GlobalTensor<T> xGm;` (naming: signature name + "Gm", e.g., `x` → `xGm`) |

## Important Notes

- **BUFFER_NUM = 2**: Fixed double-buffer mode, no other options.
- **Data Type**: Do NOT add Cast operations unless the operator specifically requires type conversion. Keep the same precision as input.
- **Multi-core Safety**: Check `if (GetBlockIdx() >= activeBlockNum) return;` if not all cores are used.

## Fixed Code Structure

```cpp
{template_code}```

## Your Task

Provide the following 7 parts:

### 1. INIT_PARAMS
Function parameters for Init (e.g., `GM_ADDR x, GM_ADDR y, uint32_t totalLength`)

**IMPORTANT**: Init must accept INDIVIDUAL parameters (GM_ADDR, uint32_t, etc.), NOT a tiling struct reference.

### 2. INIT_BODY
Body of Init function (GlobalTensor setup, pipe initialization)

### 3. PROCESS_BODY
Body of Process function (main processing loop)

### 4. PRIVATE_METHODS
**Flexible section** - define any helper methods you need:
- Common pattern: `CopyIn()`, `Compute()`, `CopyOut()`
- Complex operators: `CopyInAndCast()`, `ComputeStage1()`, etc.
- Each method needs `__aicore__ inline` prefix

Example format:
```cpp
    __aicore__ inline void CopyIn(int32_t progress) {{
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        inQueue.EnQue(xLocal);
    }}
```

### 5. MEMBER_VARS
Private member variable declarations

### 6. GLOBAL_FUNC_PARAMS
Parameters for extern "C" function (before `GM_ADDR workspace, GM_ADDR tiling`)

### 7. INIT_CALL_ARGS
Arguments passed to op.Init() (e.g., `x, y, tilingData.totalLength, tilingData.tileNum`)

## Response Format

Output each part in XML tags:
```
<init_params>...</init_params>
<init_body>...</init_body>
<process_body>...</process_body>
<private_methods>...</private_methods>
<member_vars>...</member_vars>
<global_func_params>...</global_func_params>
<init_call_args>...</init_call_args>
```

## Example (for reference only, adapt to your operator)

```cpp
// init_params: GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum

// init_body:
uint32_t blockLength = totalLength / GetBlockNum();
this->tileLength = blockLength / tileNum / BUFFER_NUM;
xGm.SetGlobalBuffer((__gm__ float*)x + blockLength * GetBlockIdx(), blockLength);
yGm.SetGlobalBuffer((__gm__ float*)y + blockLength * GetBlockIdx(), blockLength);
pipe.InitBuffer(inQueue, BUFFER_NUM, tileLength * sizeof(float));
pipe.InitBuffer(outQueue, BUFFER_NUM, tileLength * sizeof(float));

// process_body:
for (int32_t i = 0; i < tileNum * BUFFER_NUM; i++) {{
    CopyIn(i); Compute(i); CopyOut(i);
}}

// private_methods: CopyIn/Compute/CopyOut with __aicore__ inline prefix

// member_vars: TPipe, TQue, GlobalTensor, tileLength, etc.

// global_func_params: GM_ADDR x, GM_ADDR y,

// init_call_args: x, y, tilingData.totalLength, tilingData.tileNum
```

Now output all 7 parts for `{op_name}`:
"""

    # =========================================================================
    # Legacy interface (backward compatibility)
    # =========================================================================

    def get_tiling_impl_prompt(self, plan: dict, knowledge: str) -> str:
        """Legacy interface - deprecated.

        Use get_tiling_header_prompt + get_tiling_host_prompt instead.
        """
        context = {"op_name": "CustomOp", "signature": {}}
        return self.get_tiling_host_prompt(plan, context, knowledge, "")
