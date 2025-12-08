# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Default prompt implementations for Kernel + Tiling Joint Branch"""

from typing import List, Dict, Any

from .phase0 import _format_signature


# ============================================================
# Ascend Chip Specification Table
# ============================================================
# Note: UB buffer sizes are estimated values (official docs rarely disclose).
# AI Core counts are from HAMi virtualization configs and official sources.
# References:
# - https://www.theriseunion.com/en/blog/HAMi-ascend-910b-support.html
# - https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined
# - https://zhuanlan.zhihu.com/p/599049070
# ============================================================

CHIP_SPECS: Dict[str, Dict[str, Any]] = {
    "Ascend910A": {
        "ub_capacity": 256 * 1024,      # Estimated, not officially disclosed
        "ai_core_count": 30,            # 30 usable AI Cores (32 physical DaVinci Max)
        "total_cores": 32,
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "memory_gb": 32,                # 32 GB HBM
        "fp16_tflops": 256,
        "power_w": 310,
        "process": "7nm EUV (TSMC)",
        "description": "Training NPU (910A series)",
    },
    "Ascend910B2": {
        "ub_capacity": 256 * 1024,      # Estimated
        "ai_core_count": 24,            # Official: 24 AI Cores
        "ai_cpu_count": 6,
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "memory_gb": 64,                # 64 GB HBM2e
        "description": "Training NPU (910B2 series)",
    },
    "Ascend910B3": {
        "ub_capacity": 256 * 1024,      # Estimated
        "ai_core_count": 20,            # Official: 20 AI Cores
        "ai_cpu_count": 7,
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "memory_gb": 64,                # 64 GB HBM3e
        "memory_bandwidth": "1.2 TB/s",
        "description": "Training NPU (910B3 series)",
    },
    "Ascend310": {
        "ub_capacity": 128 * 1024,      # Estimated
        "ai_core_count": 1,             # Single AI Core (edge inference)
        "arm_cpu_cores": 8,             # 8x ARM A55 CPU
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "int8_tops": 16,
        "fp16_tflops": 8,
        "power_w": 8,
        "process": "12nm",
        "description": "Edge Inference NPU",
    },
    "Ascend310P": {
        "ub_capacity": 128 * 1024,      # Estimated
        "ai_core_count": 8,             # Estimated, not officially disclosed
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "description": "Inference NPU (310P series)",
    },
}

# Aliases for common usage
CHIP_SPECS["Ascend910B"] = CHIP_SPECS["Ascend910B3"]  # Default 910B -> 910B3

DEFAULT_CHIP = "Ascend910B3"


def get_chip_spec(npu_type: str) -> Dict[str, Any]:
    """Get chip specification by NPU type."""
    return CHIP_SPECS.get(npu_type, CHIP_SPECS[DEFAULT_CHIP])


def format_chip_spec(npu_type: str) -> str:
    """Format chip specification for prompt."""
    spec = get_chip_spec(npu_type)
    ub_kb = spec['ub_capacity'] // 1024
    core_count = spec.get('ai_core_count', 'unknown')
    mem_info = f", {spec['memory_gb']}GB HBM" if 'memory_gb' in spec else ""
    return f"""- Chip: {npu_type}
- UB Capacity: {ub_kb}KB per core (estimated)
- AI Core Count: {core_count}{mem_info}
- Vector Alignment: {spec['vector_align']} bytes
- Cube Tile Size: {spec['cube_m']}x{spec['cube_n']}x{spec['cube_k']}"""


# ============================================================
# Reference: Joint Branch Design (from phase1_joint_branch.md)
# ============================================================
# Why Joint Design?
# 1. Kernel depends on Tiling:
#    - Kernel needs tiling params (blockSize, tileSize) to access data
#    - Kernel loop structure is determined by tiling strategy
#
# 2. Tiling depends on Kernel:
#    - Tiling strategy needs to know kernel's compute pattern
#    - Different kernel implementations may need different tiling granularity
#
# Three-Phase Flow:
#   Phase 1: Joint Planning Discussion
#       Tiling Agent <-> Kernel Agent -> Design Consensus
#   Phase 2: Knowledge Retrieval
#       get_api_doc(), get_operator_example()
#   Phase 3: Code Implementation
#       Kernel Agent -> kernel_src
#       Tiling Agent -> tiling (or decide to use default)
#
# Default Tiling Suitable For:
#   - Add, Sub, Mul, Div (element-wise)
#   - ReLU, Sigmoid, Tanh (activations)
#   - Exp, Log, Sqrt (math functions)
#
# Custom Tiling Required For:
#   - MatMul: custom InferShape needed
#   - Reduce: need to compute reduction dimension tiling
#   - LayerNorm, Softmax: special tiling strategies
# ============================================================


class JointPromptMixin:
    """Prompts for Kernel + Tiling Joint Branch"""

    # ==================== Multi-turn Dialogue ====================

    def _extract_current_plan(self, conversation: List[dict]) -> str:
        """Extract current best plan from conversation (avoid passing full history)."""
        if not conversation:
            return None
        # Find the last tiling proposal
        for msg in reversed(conversation):
            if msg.get('role') == 'tiling' and '<response>' in msg.get('content', ''):
                content = msg['content']
                start = content.find('<response>')
                end = content.find('</response>')
                if start != -1 and end != -1:
                    return content[start:end + len('</response>')]
        return None

    def _extract_kernel_feedback(self, conversation: List[dict]) -> str:
        """Extract latest kernel feedback from conversation."""
        if not conversation:
            return None
        for msg in reversed(conversation):
            if msg.get('role') == 'kernel':
                return msg.get('content', '')
        return None

    def get_tiling_propose_prompt(self, context: dict, conversation: List[dict]) -> str:
        """Generate prompt for Tiling Specialist to propose strategy"""
        formatted_sig = _format_signature(context.get('signature'))
        compute_pattern = context.get('compute_pattern', 'other')

        # Get hardware specification from context
        npu_type = context.get('npu_type', DEFAULT_CHIP)
        hw_spec = format_chip_spec(npu_type)
        chip = get_chip_spec(npu_type)
        ub_kb = chip['ub_capacity'] // 1024
        core_count = chip.get('ai_core_count', 8)

        # Check if this is first round or revision round
        current_plan = self._extract_current_plan(conversation)
        kernel_feedback = self._extract_kernel_feedback(conversation)
        kernel_design = self._extract_kernel_design(conversation)

        # Use different prompts for first round vs revision
        if current_plan and kernel_feedback:
            return self._get_tiling_revise_prompt(
                formatted_sig, compute_pattern, context.get('python_ref'),
                current_plan, kernel_feedback, kernel_design, ub_kb, core_count
            )

        # First round: full prompt with examples
        return f"""## Role
You are the **tiling agent**. Design a tiling strategy for the kernel agent.

## Hardware
{hw_spec}

## Compute Pattern: `{compute_pattern}`

## Decision Guide

| Pattern | Strategy | Paradigm | Action |
|---------|----------|----------|--------|
| element-wise | default | vector | **Quick path** (skip analysis) |
| reduction | custom | vector | Full analysis |
| broadcast | custom | vector | Full analysis |
| matmul | custom | cube | Full analysis (cube unit) |
| other | custom | analyze | Full analysis |

## Input

### Operator Signature
{formatted_sig}

### Python Reference
```python
{context.get('python_ref')}
```

## Output

**If `element-wise`** (quick path):
<response>
Strategy: default
Paradigm: vector
block_dim: {core_count}
Reason: <1 sentence>
</response>

**Otherwise** (full analysis):
<response>
## Analysis
1. **Ops**: <operations with shapes>
2. **Dims**: <which are independent vs reduction>
3. **Memory**: <per-tile size>, fits {ub_kb}KB? <yes/no>

## Decision
- block_dim: <N> (<which dim, why>)
- tile_num: <M> (<reason>)
- buffer_num: <1|2>
- Paradigm: <vector|cube>

## Execution
```
for i in range(tile_num):
    CopyIn: <what>
    Compute: <APIs>
    CopyOut: <what>
```

## Tiling Fields
- <field>: <type> // <purpose>

## Summary
Strategy: <default|custom>, Key: <1 sentence>
</response>

## Examples

### Ex1: Add (element-wise -> quick path)
Pattern: `element-wise`, Python: `z = x + y`
<response>
Strategy: default
Paradigm: vector
block_dim: {core_count}
Reason: All dims independent, standard tiling.
</response>

### Ex2: Softmax (reduction -> full)
Pattern: `reduction`, Python: `y = softmax(x, dim=-1)` x=[B,D]
<response>
## Analysis
1. **Ops**: ReduceMax, Sub, Exp, ReduceSum, Div
2. **Dims**: B=independent (parallelize), D=reduction (full row)
3. **Memory**: D*8 bytes/row, D=1024 -> 8KB << {ub_kb}KB

## Decision
- block_dim: min({core_count}, B)
- tile_num: rowsPerCore
- buffer_num: 2
- Paradigm: vector

## Execution
```
for row in range(rowsPerCore):
    CopyIn: x[row*D:(row+1)*D]
    Compute: max, sub, exp, sum, div
    CopyOut: y[row*D:(row+1)*D]
```

## Tiling Fields
- batchSize: uint32_t // B
- featureDim: uint32_t // D
- rowsPerCore: uint32_t

## Summary
Strategy: custom, Key: Reduction along D requires full row; parallelize B.
</response>

### Ex3: MatMul (matmul -> cube)
Pattern: `matmul`, Python: `C = A @ B` A=[M,K], B=[K,N]
<response>
## Analysis
1. **Ops**: MatMul [M,K]@[K,N]->[M,N]
2. **Dims**: M,N=independent, K=reduction (accumulate)
3. **Memory**: tiles ~80KB < {ub_kb}KB

## Decision
- block_dim: min({core_count}, M/tileM)
- tile_num: nTiles * kTiles
- buffer_num: 2
- Paradigm: cube

## Execution
```
for m in myMTiles:
    for n in range(nTiles):
        acc = 0
        for k in range(kTiles):
            CopyIn: A[m,k], B[k,n]
            Compute: acc += Cube(A,B)
        CopyOut: C[m,n]
```

## Tiling Fields
- M, N, K: uint32_t
- tileM, tileN, tileK: uint32_t

## Summary
Strategy: custom, Key: Cube unit; tile all dims, accumulate K.
</response>

Now analyze the given operator:
"""

    def _extract_kernel_design(self, conversation: List[dict]) -> str:
        """Extract kernel design (including pseudocode) from conversation."""
        for msg in reversed(conversation):
            if msg.get('role') == 'kernel' and 'accepted: true' in msg.get('content', '').lower():
                content = msg['content']
                # Extract from ## Kernel Design to end of response
                start = content.find('## Kernel Design')
                if start != -1:
                    end = content.find('</response>')
                    if end != -1:
                        return content[start:end].strip()
        return None

    def _get_tiling_revise_prompt(
        self,
        formatted_sig: str,
        compute_pattern: str,
        python_ref: str,
        current_plan: str,
        kernel_feedback: str,
        kernel_design: str,
        ub_kb: int,
        core_count: int,
    ) -> str:
        """Generate concise prompt for revision round (no examples, no redundant info)."""
        # Build kernel section based on whether design exists
        if kernel_design:
            kernel_section = f"""## Kernel Design (from kernel agent)
{kernel_design}

## Kernel Feedback
{kernel_feedback}"""
        else:
            kernel_section = f"""## Kernel Feedback
{kernel_feedback}"""

        return f"""## Role
You are the **tiling agent**. Revise your tiling strategy based on kernel feedback.

The tiling fields you define will be used in the kernel pseudocode. Ensure consistency.

## Context
- Pattern: `{compute_pattern}`
- UB: {ub_kb}KB, Cores: {core_count}

## Operator Signature
{formatted_sig}

## Python Reference
```python
{python_ref}
```

## Your Previous Plan
{current_plan}

{kernel_section}

## Task
Revise the plan to address the feedback. Use the **same format** as your previous plan.

**If quick path** (element-wise):
<response>
Strategy: default
Paradigm: vector
block_dim: {core_count}
Reason: <1 sentence>
</response>

**If full analysis**:
<response>
## Analysis
1. **Ops**: ...
2. **Dims**: ...
3. **Memory**: ...

## Decision
- block_dim: ...
- tile_num: ...
- buffer_num: ...
- Paradigm: ...

## Execution
```
...
```

## Tiling Fields
- ...

## Summary
Strategy: ..., Key: ...
</response>
"""

    def _extract_tiling_strategy(self, conversation: List[dict]) -> dict:
        """Extract tiling strategy info from the current plan."""
        current_plan = self._extract_current_plan(conversation)
        if not current_plan:
            return {"strategy": "unknown", "paradigm": "vector"}

        plan_lower = current_plan.lower()
        strategy = "default" if "strategy: default" in plan_lower else "custom"
        paradigm = "cube" if "paradigm: cube" in plan_lower else "vector"
        return {"strategy": strategy, "paradigm": paradigm}

    def get_kernel_review_prompt(self, context: dict, conversation: List[dict]) -> str:
        """Generate prompt for Kernel Specialist to review the proposal"""
        formatted_sig = _format_signature(context.get('signature'))
        compute_pattern = context.get('compute_pattern', 'other')

        # Get hardware specification
        npu_type = context.get('npu_type', DEFAULT_CHIP)
        hw_spec = format_chip_spec(npu_type)

        # Extract current tiling plan and check if revision round
        current_plan = self._extract_current_plan(conversation)
        previous_feedback = None

        # Check if this is a revision round (kernel already gave feedback before)
        for msg in conversation:
            if msg.get('role') == 'kernel':
                previous_feedback = msg.get('content', '')
                break

        if previous_feedback and current_plan:
            return self._get_kernel_re_review_prompt(
                formatted_sig, compute_pattern, context.get('python_ref'),
                current_plan, previous_feedback
            )

        # First round: full prompt with examples
        return f"""## Role
You are the **kernel agent**. Review the tiling proposal and design kernel implementation.

**This is conceptual design phase.** You describe:
- Operations conceptually (e.g., "row-wise reduction"), not exact API names
- Useful references for knowledge retrieval (similar ops, APIs to look up)

The retrieval system will fetch actual API docs and examples based on your output.

## Hardware
{hw_spec}

## Compute Pattern: `{compute_pattern}`

## Operator Signature
{formatted_sig}

## Python Reference
```python
{context.get('python_ref')}
```

## Tiling Proposal
{current_plan}

## Review Checklist
1. **Paradigm match**: vector for element-wise/reduction/broadcast, cube for matmul
2. **Memory fit**: tile size fits UB capacity?
3. **Dim handling**: reduction dims handled correctly? independent dims parallelized?
4. **Alignment**: cube tiles aligned to 16? vector aligned to 32?

## Your Tasks
1. Review tiling strategy (reject if checklist fails)
2. Design kernel data flow: CopyIn -> Compute -> CopyOut
3. List operations (conceptual description)
4. List useful references for retrieval

## Decision Guide

| Paradigm | When to Use | Pipeline | Typical Operations |
|----------|-------------|----------|-------------------|
| vector | element-wise, reduction, broadcast | double_buffer | add, sub, mul, exp, reduce-sum, reduce-max |
| cube | matmul | double_buffer | matrix multiply-accumulate |

## Output Format

**If you ACCEPT:**
<response>
accepted: true

## Kernel Design
- Pipeline: <single_buffer | double_buffer>
- Operations: [<conceptual op1>, <conceptual op2>, ...]

## Kernel Pseudocode
```cpp
// Using tiling fields: <field1>, <field2>, ...
for (...) {
    // CopyIn
    <load data using tiling fields>

    // Compute
    <operations using tiling fields>

    // CopyOut
    <store data using tiling fields>
}
```

## Useful References
- <name>: <why>
</response>

**If you REJECT:**
<response>
accepted: false

## Issues
1. <which checklist item failed, why>

## Suggestions
<specific changes for tiling agent>
</response>

## Examples

### Ex1: Accept element-wise Add (default tiling)
Pattern: `element-wise`, Tiling: default, vector, fields: totalLength, tileNum, tileLength
<response>
accepted: true

## Kernel Design
- Pipeline: double_buffer
- Operations: [element-wise add]

## Kernel Pseudocode
```cpp
// Using tiling fields: totalLength, tileNum, tileLength
for (int i = 0; i < tileNum; i++) {
    // CopyIn
    xLocal = LoadTile(xGm, i * tileLength, tileLength);
    yLocal = LoadTile(yGm, i * tileLength, tileLength);

    // Compute
    zLocal = Add(xLocal, yLocal, tileLength);

    // CopyOut
    StoreTile(zGm, i * tileLength, zLocal, tileLength);
}
```

## Useful References
- add_custom: similar element-wise pattern
</response>

### Ex2: Accept Softmax (custom tiling)
Pattern: `reduction`, Tiling: custom, vector, fields: batchSize, featureDim, rowsPerCore
<response>
accepted: true

## Kernel Design
- Pipeline: double_buffer
- Operations: [row-wise max, broadcast sub, element-wise exp, row-wise sum, broadcast div]

## Kernel Pseudocode
```cpp
// Using tiling fields: batchSize, featureDim, rowsPerCore
for (int row = 0; row < rowsPerCore; row++) {
    int offset = (blockIdx * rowsPerCore + row) * featureDim;

    // CopyIn: load one row
    xLocal = LoadTile(xGm, offset, featureDim);

    // Compute
    maxVal = ReduceMax(xLocal, featureDim);
    xLocal = Sub(xLocal, maxVal, featureDim);      // broadcast
    xLocal = Exp(xLocal, featureDim);
    sumVal = ReduceSum(xLocal, featureDim);
    xLocal = Div(xLocal, sumVal, featureDim);      // broadcast

    // CopyOut
    StoreTile(yGm, offset, xLocal, featureDim);
}
```

## Useful References
- softmax_custom: similar reduction pattern
- ReduceMax: need for row-wise max
- ReduceSum: need for normalization
</response>

### Ex3: Reject wrong paradigm
Pattern: `matmul`, Tiling: custom, **vector** (wrong!)
<response>
accepted: false

## Issues
1. Paradigm mismatch: matmul requires cube, not vector
2. Alignment: cube tiles must be aligned to 16

## Suggestions
Change paradigm to cube. Use tileM/tileN/tileK as multiples of 16.
</response>

Now review the tiling proposal. Output ONLY the `<response>` block:
"""

    def _get_kernel_re_review_prompt(
        self,
        formatted_sig: str,
        compute_pattern: str,
        python_ref: str,
        current_plan: str,
        previous_feedback: str,
    ) -> str:
        """Generate concise prompt for re-reviewing revised tiling proposal."""
        return f"""## Role
You are the **kernel agent**. Re-review the revised tiling proposal.

**Conceptual design phase** - describe operations conceptually, provide pseudocode using tiling fields.

## Context
- Pattern: `{compute_pattern}`

## Operator Signature
{formatted_sig}

## Python Reference
```python
{python_ref}
```

## Revised Tiling Proposal
{current_plan}

## Your Previous Feedback
{previous_feedback}

## Task
Check if the revised proposal addresses your feedback. Use the **same format** as before.

**If you ACCEPT:**
<response>
accepted: true

## Kernel Design
- Pipeline: <...>
- Operations: [...]

## Kernel Pseudocode
```cpp
// Using tiling fields: <fields from tiling proposal>
<pseudocode using those fields>
```

## Useful References
- <name>: <why>
</response>

**If still needs revision:**
<response>
accepted: false

## Issues
1. <remaining issue>

## Suggestions
<what to change>
</response>
"""

    # ==================== Code Implementation ====================

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
