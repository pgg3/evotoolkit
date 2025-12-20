# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Kernel Agent Prompts for Joint Branch

This module contains all prompts for the Kernel Specialist agent:
- First round review prompt (with examples)
- Re-review prompt for revision rounds (concise)
- Final round prompt (must produce implementable design)
"""

from typing import List

from evotoolkit.task.cann_init.method_interface.prompts.phase0 import _format_signature_for_kernel
from .chip_specs import DEFAULT_CHIP, format_chip_spec
from .utils import extract_current_plan


class KernelPromptsMixin:
    """Prompts for the Kernel Specialist agent"""

    def get_kernel_review_prompt(
        self, context: dict, conversation: List[dict], is_final_round: bool = False
    ) -> str:
        """Generate prompt for Kernel Specialist to review the proposal"""
        formatted_sig = _format_signature_for_kernel(context.get('signature'))

        # Get hardware specification
        npu_type = context.get('npu_type', DEFAULT_CHIP)
        hw_spec = format_chip_spec(npu_type)

        # Extract current tiling plan and check if revision round
        current_plan = extract_current_plan(conversation)
        previous_feedback = None

        # Check if this is a revision round (kernel already gave feedback before)
        # Use reversed() to get the LATEST kernel feedback, not the first one
        for msg in reversed(conversation):
            if msg.get('role') == 'kernel':
                previous_feedback = msg.get('content', '')
                break

        # Final round: must produce implementable design regardless of issues
        if is_final_round:
            return self._get_kernel_final_round_prompt(
                formatted_sig, context.get('python_ref'),
                current_plan, previous_feedback, hw_spec
            )

        if previous_feedback and current_plan:
            return self._get_kernel_re_review_prompt(
                formatted_sig, context.get('python_ref'),
                current_plan, previous_feedback
            )

        # First round: full prompt with examples
        return self._get_kernel_first_round_prompt(
            formatted_sig, context.get('python_ref'),
            current_plan, hw_spec
        )

    def _get_kernel_first_round_prompt(
        self,
        formatted_sig: str,
        python_ref: str,
        current_plan: str,
        hw_spec: str,
    ) -> str:
        """Generate full prompt for first round review (with examples)."""
        return f"""## Role
You are the **Kernel Agent** in a multi-agent Ascend C code generation pipeline.

Your task: Review the tiling proposal and design kernel implementation strategy.

**This is the conceptual design phase.** You will NOT write actual code.

---

## Hardware
{hw_spec}

## Operator Signature
{formatted_sig}

## Python Reference
```python
{python_ref}
```

## Tiling Proposal
{current_plan}

---

## Review Focus

1. **Alignment**: For matmul, tiles align to fractal block (16x16 for fp16, varies by dtype)
2. **Multi-core**: Offset computed via `GetBlockIdx()`? (NOT in tiling fields)

---

## Common Mistakes

- [X] **Don't require block_offset** -> each core computes its own offset via GetBlockIdx()
- [X] **Don't forget to confirm Tiling Fields** -> list what kernel actually needs

---

## Output Format

<response>
## Reasoning
<1-2 sentences: Is the tiling proposal correct? What's the kernel strategy?>

accepted: <true | false>
strategy: <default | generate>

(If strategy=default: use default template, NO custom tiling fields needed)
(If strategy=generate: you MUST provide Tiling Fields Required below)

(If accepted=true, add:)

## Kernel Design
- Pipeline: <double_buffer>
- Operations: [op1, op2, ...]

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
for (...) {{
    CopyIn: <load data>
    Compute: <operations>
    CopyOut: <store data>
}}
```

(If strategy=generate, add:)

## Tiling Fields Required
- <field>: <type> // <purpose>

## Useful References
- APIs: [<API names for doc lookup>]
- Examples: [<similar operators>]

(If accepted=false, add:)

## Issues
1. <what's wrong>

## Suggestions
<how to fix>
</response>

---

## Examples

### Ex1: Accept ReLU (default strategy)
Tiling: default (single input, same output shape)
<response>
## Reasoning
ReLU is single-input element-wise. Default template is appropriate.

accepted: true
strategy: default

## Kernel Design
- Pipeline: double_buffer
- Operations: [ReLU]

## Kernel Pseudocode
```cpp
// Default template: totalLength, tileNum from TilingData
// Multi-core: GetBlockIdx() for core offset
for (int i = 0; i < tileNum; i++) {{
    offset = GetBlockIdx() * tilesPerCore + i
    CopyIn: x[offset * tileLength : (offset+1) * tileLength]
    Compute: y = Relu(x)
    CopyOut: y[offset * tileLength : (offset+1) * tileLength]
}}
```

## Useful References
- APIs: [Relu, DataCopy]
- Examples: [relu_custom]
</response>

### Ex2: Accept Add (generate strategy)
Tiling: generate (two inputs)
<response>
## Reasoning
Add has two inputs, needs generate strategy. Tiling proposal is correct.

accepted: true
strategy: generate

## Kernel Design
- Pipeline: double_buffer
- Operations: [element-wise add]

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
for (int i = 0; i < tileNum; i++) {{
    globalIdx = GetBlockIdx() * tileNum + i
    CopyIn: x[globalIdx * tileLength], y[globalIdx * tileLength]
    Compute: z = Add(x, y)
    CopyOut: z[globalIdx * tileLength]
}}
```

## Tiling Fields Required
- totalLength: uint32_t // total elements
- tileNum: uint32_t // tiles per core
- tileLength: uint32_t // elements per tile

## Useful References
- APIs: [Add]
- Examples: [add_custom]
</response>

### Ex3: Accept Softmax (generate strategy)
Tiling: generate (row-wise reduction)
<response>
## Reasoning
Softmax needs row-wise reduction. Generate strategy required.

accepted: true
strategy: generate

## Kernel Design
- Pipeline: double_buffer
- Operations: [ReduceMax, Sub, Exp, ReduceSum, Div]

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
for (int row = 0; row < rowsPerCore; row++) {{
    globalRow = GetBlockIdx() * rowsPerCore + row
    CopyIn: x[globalRow * featureDim : (globalRow+1) * featureDim]
    Compute: max -> sub -> exp -> sum -> div
    CopyOut: y[globalRow * featureDim : (globalRow+1) * featureDim]
}}
```

## Tiling Fields Required
- batchSize: uint32_t // total rows
- featureDim: uint32_t // row length
- rowsPerCore: uint32_t // rows per core

## Useful References
- APIs: [ReduceMax, ReduceSum, Exp, Sub, Div]
- Examples: [softmax_custom]
</response>

### Ex4: Accept MatMul (matrix multiplication)
Tiling: generate (matmul with tile alignment)
<response>
## Reasoning
MatMul tiling proposal is correct. Tile sizes aligned to fractal block.

accepted: true
strategy: generate

## Kernel Design
- Pipeline: double_buffer
- Operations: [matrix multiply-accumulate]

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for M-tile assignment
for m_tile, n_tile, k_tile:
    CopyIn: A[m,k], B[k,n]
    Compute: C += Mmad(A, B)
    CopyOut: C[m,n]
```

## Tiling Fields Required
- M, N, K: uint32_t // matrix dimensions
- tileM, tileN, tileK: uint32_t // tile sizes (aligned to fractal block)

## Useful References
- APIs: [Mmad, MatMul]
- Examples: [matmul_custom]
</response>

---

Now review the tiling proposal:
"""

    def _get_kernel_re_review_prompt(
        self,
        formatted_sig: str,
        python_ref: str,
        current_plan: str,
        previous_feedback: str,
    ) -> str:
        """Generate concise prompt for re-reviewing revised tiling proposal."""
        return f"""## Role
You are the **Kernel Agent**. Re-review the revised tiling proposal.

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
Check if the revised proposal addresses your feedback. Use the **same format**:

<response>
## Reasoning
<Does the revision address your feedback? Any remaining issues?>

accepted: <true | false>
strategy: <default | generate>

(If accepted=true:)
## Kernel Design
...

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
...
```

(If strategy=generate:)
## Tiling Fields Required
...

## Useful References
...

(If accepted=false:)
## Issues
...

## Suggestions
...
</response>
"""

    def _get_kernel_final_round_prompt(
        self,
        formatted_sig: str,
        python_ref: str,
        current_plan: str,
        previous_feedback: str,
        hw_spec: str,
    ) -> str:
        """Generate prompt for final round - MUST produce implementable design."""
        feedback_section = ""
        if previous_feedback:
            feedback_section = f"""## Your Previous Feedback
{previous_feedback}

"""
        return f"""## Role
You are the **Kernel Agent**. This is the **FINAL ROUND**.

## CRITICAL: You MUST produce an implementable design now.

You cannot reject. Either:
1. Accept if workable, OR
2. Accept with modifications - fix issues yourself

**Do NOT output "accepted: false"**

---

## Hardware
{hw_spec}

## Operator Signature
{formatted_sig}

## Python Reference
```python
{python_ref}
```

## Current Tiling Proposal
{current_plan}

{feedback_section}## Reminder
- Each core computes offset via `GetBlockIdx()` (NOT in tiling fields)
- Choose strategy: default (single input, same output shape) or generate (all other cases)

---

## Output Format

<response>
## Reasoning
<Any assumptions or workarounds needed? If none, write "Proposal is correct.">

accepted: true
strategy: <default | generate>

## Kernel Design
- Pipeline: <double_buffer>
- Operations: [op1, op2, ...]

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
for (...) {{
    CopyIn: ...
    Compute: ...
    CopyOut: ...
}}
```

## Tiling Execution
```
for <loop>:
    CopyIn: <data>
    Compute: <ops>
    CopyOut: <data>
```

(If strategy=generate:)
## Tiling Fields Required
- <field>: <type> // <purpose>

## Useful References
- APIs: [...]
- Examples: [...]
</response>

This design will be used for code generation. Make it complete!
"""
