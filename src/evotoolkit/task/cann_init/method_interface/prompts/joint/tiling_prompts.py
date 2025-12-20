# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tiling Agent Prompts for Joint Branch

This module contains all prompts for the Tiling Specialist agent:
- First round proposal prompt (with examples)
- Revision round prompt (concise, no examples)
"""

from typing import List

from evotoolkit.task.cann_init.method_interface.prompts.phase0 import _format_signature_for_kernel
from .chip_specs import DEFAULT_CHIP, get_chip_spec, format_chip_spec
from .utils import extract_current_plan, extract_kernel_feedback, extract_kernel_design


class TilingPromptsMixin:
    """Prompts for the Tiling Specialist agent"""

    def get_tiling_propose_prompt(self, context: dict, conversation: List[dict]) -> str:
        """Generate prompt for Tiling Specialist to propose strategy"""
        formatted_sig = _format_signature_for_kernel(context.get('signature'))

        # Get hardware specification from context
        npu_type = context.get('npu_type', DEFAULT_CHIP)
        hw_spec = format_chip_spec(npu_type)
        chip = get_chip_spec(npu_type)
        ub_kb = chip['ub_capacity'] // 1024
        core_count = chip.get('ai_core_count', 8)

        # Check if this is first round or revision round
        current_plan = extract_current_plan(conversation)
        kernel_feedback = extract_kernel_feedback(conversation)
        kernel_design = extract_kernel_design(conversation)

        # Use different prompts for first round vs revision
        if current_plan and kernel_feedback:
            return self._get_tiling_revise_prompt(
                formatted_sig, context.get('python_ref'),
                current_plan, kernel_feedback, kernel_design, ub_kb, core_count
            )

        # First round: full prompt with examples
        return f"""## Role
You are the **Tiling Agent** in a multi-agent Ascend C code generation pipeline.

Your task: Design the tiling strategy for an NPU operator.

**This is the conceptual design phase.** You will NOT write code. Your design will be used by the Kernel Agent.

---

## What is Tiling?

Tiling divides large data into smaller blocks (tiles) that fit in the NPU's on-chip memory.

**Key concepts:**
- UB (Unified Buffer): **{ub_kb}KB per core** - tiles must fit here
- Multiple cores ({core_count} cores) process tiles in parallel
- **Tiling Fields**: Parameters computed on Host, passed to Kernel via TilingData struct

---

## Hardware
{hw_spec}

---

## Strategy Decision

### Strategy: `default`
Use when: **Single tensor input AND output shape = input shape**

The default template provides:
- Tiling Fields: `{{totalLength, tileNum}}`
- Logic: Flatten all dims, divide evenly across cores

### Strategy: `generate`
Use for: **All other cases** (multiple inputs, shape changes, reductions, etc.)

You design custom Tiling Fields and Execution Flow.

**Paradigm Selection:**
- **vector**: Element-wise, reduction, broadcast (uses Vector Unit)
- **cube**: Matrix multiplication only (uses Cube Unit, tiles align to 16)

---

## Operator Signature

{formatted_sig}

**About this signature:**
- Extracted from Python reference (`get_inputs()` and `get_init_inputs()`)
- **Tensor Inputs/Outputs**: Data that flows through kernel, needs tiling
- **Attrs**: Scalar parameters, include as tiling fields (passed via TilingData)

### Python Reference
```python
{context.get('python_ref')}
```

---

## Common Mistakes

- [X] **Don't include per-core values** (like `block_offset`) in Tiling Fields
  -> Tiling data is shared by ALL cores. Each core computes offset via `GetBlockIdx()`
- [X] **Don't forget Attrs** -> Include scalar parameters as tiling fields

---

## Output Format

<response>
## Reasoning
<1-2 sentences: What does this operator do? Single tensor input with same output shape? If not, why generate?>

Strategy: <default | generate>

(If generate, add:)

## Tiling Design
- Paradigm: <vector | cube>
- block_dim: <number of cores>

## Tiling Fields
- <field>: <type> // <purpose>
- ...

## Execution Flow
```
for <loop over tiles>:
    CopyIn: <what data to load, with index range>
    CopyOut: <what data to store, with index range>
```
Note: <why this tiling pattern? e.g., "full row needed for reduction">
</response>

---

## Examples

### Example 1: Exp (default)
Signature: x: float16 → y: float16 (same shape)
<response>
## Reasoning
Exp applies element-wise to single tensor. Single input, same output shape.

Strategy: default
</response>

### Example 2: Add (generate - but simple)
Signature: x: float16, y: float16 → z: float16 (same shape)
<response>
## Reasoning
Add has two tensor inputs, so not single-input. Use generate, but logic is simple element-wise.

Strategy: generate

## Tiling Design
- Paradigm: vector
- block_dim: {core_count}

## Tiling Fields
- totalLength: uint32_t // total elements
- tileNum: uint32_t // number of tiles
- tileLength: uint32_t // elements per tile

## Execution Flow
```
for tile in range(tilesPerCore):
    CopyIn: x[tile_start : tile_end], y[tile_start : tile_end]
    CopyOut: z[tile_start : tile_end]
```
Note: Element-wise, tiles are independent.
</response>

### Example 3: Softmax (generate - reduction)
Signature: x: float16[B, D] → y: float16[B, D]
<response>
## Reasoning
Softmax needs full row for normalization (reduction along D). Same shape but requires custom tiling.

Strategy: generate

## Tiling Design
- Paradigm: vector
- block_dim: min({core_count}, B)

## Tiling Fields
- batchSize: uint32_t // total rows (B)
- featureDim: uint32_t // row length (D)
- rowsPerCore: uint32_t // rows per core

## Execution Flow
```
for row in range(rowsPerCore):
    CopyIn: x[row * D : (row+1) * D]
    CopyOut: y[row * D : (row+1) * D]
```
Note: Full row needed for softmax reduction along D.
</response>

### Example 4: MatMul (generate - cube)
Signature: A: float16[M,K], B: float16[K,N] → C: float16[M,N]
<response>
## Reasoning
MatMul has two inputs and output shape differs. Requires cube paradigm.

Strategy: generate

## Tiling Design
- Paradigm: cube
- block_dim: min({core_count}, M/tileM)

## Tiling Fields
- M: uint32_t // rows of A/C
- N: uint32_t // cols of B/C
- K: uint32_t // reduction dim
- tileM: uint32_t // aligned to 16
- tileN: uint32_t // aligned to 16
- tileK: uint32_t // aligned to 16

## Execution Flow
```
for m_tile in assigned_M_tiles:
    for n_tile in range(N_tiles):
        for k_tile in range(K_tiles):
            CopyIn: A[m_tile, k_tile], B[k_tile, n_tile]
        CopyOut: C[m_tile, n_tile]
```
Note: K dimension is reduction (accumulate across k_tiles).
</response>

---

Now analyze the given operator:
"""

    def _get_tiling_revise_prompt(
        self,
        formatted_sig: str,
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
            kernel_section = f"""## Kernel Design (from Kernel Agent)
{kernel_design}

## Kernel Feedback
{kernel_feedback}"""
        else:
            kernel_section = f"""## Kernel Feedback
{kernel_feedback}"""

        return f"""## Role
You are the **Tiling Agent**. Revise your tiling strategy based on Kernel Agent feedback.

## Hardware Context
- UB Capacity: {ub_kb}KB per core
- AI Cores: {core_count}

## Operator
{formatted_sig}

```python
{python_ref}
```

## Your Previous Plan
{current_plan}

{kernel_section}

## Task
Address the feedback and revise. Use the **same format**:

<response>
## Reasoning
<What changes did you make based on feedback?>

Strategy: <default | generate>

(If generate:)
## Tiling Design
...

## Tiling Fields
...

## Execution Flow
...
</response>
"""
