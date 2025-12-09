# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tiling Agent Prompts for Joint Branch

This module contains all prompts for the Tiling Specialist agent:
- First round proposal prompt (with examples)
- Revision round prompt (concise, no examples)
"""

from typing import List

from evotoolkit.task.cann_init.method_interface.prompts.phase0 import _format_signature
from .chip_specs import DEFAULT_CHIP, get_chip_spec, format_chip_spec
from .utils import extract_current_plan, extract_kernel_feedback, extract_kernel_design


class TilingPromptsMixin:
    """Prompts for the Tiling Specialist agent"""

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
        current_plan = extract_current_plan(conversation)
        kernel_feedback = extract_kernel_feedback(conversation)
        kernel_design = extract_kernel_design(conversation)

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
