# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tiling Agent Prompts for Joint Branch

This module contains all prompts for the Tiling Specialist agent:
- First round proposal prompt (with examples)
- Revision round prompt (concise, no examples)
"""

from typing import List

from evotoolkit.task.cann_init.method_interface.cann_initer.prompts.phase0 import _format_signature_for_kernel
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
        return f"""## 1. Role

You are the **Tiling Agent** in a multi-agent Ascend C code generation pipeline.

**Your task**: Design the tiling strategy for an NPU operator.

This is the conceptual design phase. You will NOT write code. Your design will be reviewed by the Kernel Agent.

---

## 2. Background

### 2.1 Ascend C Architecture

| Component | Role |
|-----------|------|
| **Host (CPU)** | Computes tiling parameters, launches kernel |
| **Kernel (NPU)** | Executes computation on tiles |
| **TilingData** | Struct that passes parameters from Host to Kernel |

### 2.2 What is Tiling?

Tiling divides large data into smaller blocks (tiles) that fit in the NPU's on-chip memory (UB).

**What is a tile?** A chunk of data small enough to fit in UB. Shape depends on access pattern:
- **Element-wise** (Add, ReLU): 1D contiguous segment
- **Row-wise** (Softmax): complete row(s) for reduction
- **Block-wise** (MatMul): 2D rectangular block

### 2.3 Key Concepts

| Concept | Description |
|---------|-------------|
| **UB (Unified Buffer)** | {ub_kb}KB per core - all tiles must fit here |
| **Multi-core** | {core_count} cores process tiles in parallel. Each core computes its offset via `GetBlockIdx()` |
| **Double Buffer** | Pipeline optimization: while computing tile N, prefetch tile N+1. Requires **2× buffer space** |
| **Tiling Fields** | Parameters defined in TilingData, directly accessible in kernel |

### 2.4 Hardware

{hw_spec}

---

## 3. Your Task

Analyze this operator and design its tiling strategy.

### Operator Signature

{formatted_sig}

- **Tensor Inputs/Outputs**: Data that flows through kernel, needs tiling
- **Attrs**: Scalar parameters, must include in Tiling Fields

### Python Reference

Read this code to understand the operator's computation pattern:

```python
{context.get('python_ref')}
```

---

## 4. Design Flow

### Step 1: Choose Strategy

| Strategy | When to Use |
|----------|-------------|
| `default` | **Single tensor input** AND **output shape = input shape** |
| `generate` | All other cases (multiple inputs, shape changes, reductions, etc.) |

Default template provides: `{{totalLength, tileNum}}`, flatten all dims, divide evenly.

### Step 2: Design Tiling (if `generate`)

1. **Identify access pattern**: element-wise / row-wise / block-wise
2. **Set block_dim**: number of cores to use (≤ {core_count})
3. **Define Tiling Fields**: parameters kernel needs (include all Attrs)
4. **Design Execution Flow**: how each core iterates over its tiles

### Step 3: Verify Memory

Use this formula:
```
Total UB = Σ (tile_size_per_tensor × 2)   // 2× for double buffer

where tile_size = elements_per_tile × bytes_per_element
```

Ensure Total UB < {ub_kb} KB. If not, reduce tile size.

---

## 5. Guidelines

### Tiling Fields

- Use `uint32_t` for counts, lengths, dimensions (most common)
- Use `int32_t` only if values can be negative (e.g., axis=-1)
- Field names: camelCase (e.g., `tileLength`, `rowsPerCore`)
- **Must include all Attrs** from signature

### Common Mistakes

- ✗ Including per-core values (like `blockOffset`) in Tiling Fields
  → Tiling data is shared by ALL cores. Each core computes offset via `GetBlockIdx()`
- ✗ Forgetting Attrs
  → Always include scalar parameters from signature

### Edge Case

- If data doesn't divide evenly, last tile handles remainder (kernel uses `min(tileLength, remaining)`)

---

## 6. Output Format

<response>
## Reasoning
<1-2 sentences: What does this operator do? What's its access pattern? Why this strategy?>

Strategy: <default | generate>

(If generate, continue with:)

## Tiling Design
- block_dim: <number of cores to use>

## Tiling Fields
- <field>: <type> // <purpose>
- ...

## Execution Flow
```
for <loop over tiles>:
    CopyIn: <what data, index range>
    CopyOut: <what data, index range>
```
Note: <why this pattern>

## Memory Check
- <tensor>: <elements> × <bytes> = X KB (× 2 for double buffer)
- ...
- Total: Y KB < {ub_kb} KB ✓
</response>

---

## 7. Examples

### Example 1: Exp (default strategy)
**Signature**: x: float16 → y: float16 (same shape)

<response>
## Reasoning
Exp is element-wise on single tensor. Single input, same output shape → default.

Strategy: default
</response>

### Example 2: Add (generate - element-wise)
**Signature**: x: float16, y: float16 → z: float16 (same shape)

<response>
## Reasoning
Add is element-wise but has two inputs → generate. Access pattern: 1D contiguous.

Strategy: generate

## Tiling Design
- block_dim: {core_count}

## Tiling Fields
- totalLength: uint32_t // total elements
- tileNum: uint32_t // tiles per core
- tileLength: uint32_t // elements per tile

## Execution Flow
```
for tile in range(tileNum):
    CopyIn: x[start:end], y[start:end]
    CopyOut: z[start:end]
```
Note: Element-wise, tiles are independent.

## Memory Check
- x, y, z: 8192 × 2B = 16KB each (× 2 = 32KB)
- Total: 32KB × 3 = 96KB < {ub_kb}KB ✓
</response>

### Example 3: Softmax (generate - row-wise reduction)
**Signature**: x: float16[B, D] → y: float16[B, D]

<response>
## Reasoning
Softmax needs full row for reduction along D. Access pattern: row-wise.

Strategy: generate

## Tiling Design
- block_dim: min({core_count}, B)

## Tiling Fields
- batchSize: uint32_t // total rows (B)
- featureDim: uint32_t // row length (D)
- rowsPerCore: uint32_t // rows assigned to each core

## Execution Flow
```
for row in range(rowsPerCore):
    CopyIn: x[row, 0:D]  // full row
    CopyOut: y[row, 0:D]
```
Note: Full row required for max/sum reduction.

## Memory Check
- x row: 4096 × 2B = 8KB (× 2 = 16KB)
- y row: 8KB (× 2 = 16KB)
- tmp (max, sum): 2 scalars, negligible
- Total: ~32KB < {ub_kb}KB ✓
</response>

### Example 4: MatMul (generate - block-wise)
**Signature**: A: float16[M,K], B: float16[K,N] → C: float16[M,N]

<response>
## Reasoning
MatMul has two inputs, output shape differs. Access pattern: 2D blocks aligned to fractal (16×16 for fp16).

Strategy: generate

## Tiling Design
- block_dim: min({core_count}, M/tileM)

## Tiling Fields
- M, N, K: uint32_t // matrix dimensions
- tileM, tileN, tileK: uint32_t // tile sizes (aligned to 16)

## Execution Flow
```
for m_tile in assigned_M_tiles:
    for n_tile in range(N/tileN):
        for k_tile in range(K/tileK):
            CopyIn: A[m, k], B[k, n]
        CopyOut: C[m, n]
```
Note: K is reduction dim, accumulate across k_tiles.

## Memory Check
- A tile: 64×64×2B = 8KB (× 2 = 16KB)
- B tile: 8KB (× 2 = 16KB)
- C tile: 8KB (× 2 = 16KB)
- Total: 48KB < {ub_kb}KB ✓
</response>

### Example 5: Clamp with Attrs
**Signature**: x: float16 → y: float16, Attrs: min_val: float, max_val: float

<response>
## Reasoning
Clamp is element-wise on single tensor, same output shape. But has Attrs that must be included in Tiling Fields.

Strategy: default

(Note: default strategy works, but Attrs must be added to TilingData)

## Tiling Fields (additional)
- minVal: float // clamp lower bound
- maxVal: float // clamp upper bound
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

---

## Context

**Hardware**: UB = {ub_kb}KB per core, {core_count} cores, Double Buffer = 2×

**Operator**:
{formatted_sig}

```python
{python_ref}
```

---

## Your Previous Plan

{current_plan}

---

{kernel_section}

---

## Task

Address the feedback and revise. Output format:

<response>
## Reasoning
<What changes did you make based on feedback?>

Strategy: <default | generate>

(If generate:)

## Tiling Design
- block_dim: ...

## Tiling Fields
- <field>: <type> // <purpose>

## Execution Flow
```
for ...:
    CopyIn: ...
    CopyOut: ...
```

## Memory Check
- <tensor>: <elements> × <bytes> = X KB (× 2 for double buffer)
- Total: Y KB < {ub_kb} KB ✓
</response>
"""
