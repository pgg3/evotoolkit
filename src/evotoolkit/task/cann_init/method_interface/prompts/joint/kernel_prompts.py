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

1. **Paradigm**: vector (element-wise, reduction, broadcast) or cube (matmul)?
2. **Multi-core**: Offset computed via `GetBlockIdx()`? (NOT in tiling fields)

---

## Common Mistakes

- [X] **Don't accept wrong paradigm** -> matmul MUST use cube, not vector
- [X] **Don't require block_offset** -> each core computes its own offset via GetBlockIdx()
- [X] **Don't forget to confirm Tiling Fields** -> list what kernel actually needs

---

## Output Format

<response>
## Reasoning
<1-2 sentences: Is the tiling proposal correct? What's the kernel strategy?>

accepted: <true | false>

(If true, add:)

## Kernel Design
- Paradigm: <vector | cube>
- Pipeline: <double_buffer>
- Operations: [op1, op2, ...]

## Kernel Pseudocode
```cpp
// Using tiling fields: <fields>
for (...) {{
    CopyIn: <load data>
    Compute: <operations>
    CopyOut: <store data>
}}
```

## Tiling Fields Required
- <field>: <type> // <purpose>

## Useful References
- APIs: [<API names for doc lookup>]
- Examples: [<similar operators>]

(If false, add:)

## Issues
1. <what's wrong>

## Suggestions
<how to fix>
</response>

---

## Examples

### Ex1: Accept Add
Tiling: generate, Paradigm: vector
<response>
## Reasoning
Simple element-wise add. Tiling proposal is correct with vector paradigm.

accepted: true

## Kernel Design
- Paradigm: vector
- Pipeline: double_buffer
- Operations: [element-wise add]

## Kernel Pseudocode
```cpp
// Using tiling fields: totalLength, tileNum, tileLength
for (int i = 0; i < tileNum; i++) {{
    CopyIn: x[i], y[i]
    Compute: z = Add(x, y)
    CopyOut: z[i]
}}
```

## Tiling Fields Required
- totalLength: uint32_t // total elements
- tileNum: uint32_t // number of tiles
- tileLength: uint32_t // elements per tile

## Useful References
- APIs: [Add]
- Examples: [add_custom]
</response>

### Ex2: Accept Softmax
Tiling: generate, Paradigm: vector, fields: batchSize, featureDim, rowsPerCore
<response>
## Reasoning
Softmax needs row-wise reduction. Tiling correctly processes full rows. Vector paradigm is appropriate.

accepted: true

## Kernel Design
- Paradigm: vector
- Pipeline: double_buffer
- Operations: [ReduceMax, Sub, Exp, ReduceSum, Div]

## Kernel Pseudocode
```cpp
// Using tiling fields: featureDim, rowsPerCore
for (int row = 0; row < rowsPerCore; row++) {{
    offset = (GetBlockIdx() * rowsPerCore + row) * featureDim
    CopyIn: x[offset : offset + featureDim]
    Compute: max -> sub -> exp -> sum -> div
    CopyOut: y[offset : offset + featureDim]
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

### Ex3: Accept MatMul (cube)
Tiling: generate, Paradigm: cube
<response>
## Reasoning
MatMul requires cube paradigm for matrix multiply. Tiling proposal is correct.

accepted: true

## Kernel Design
- Paradigm: cube
- Pipeline: double_buffer
- Operations: [matrix multiply-accumulate]

## Kernel Pseudocode
```cpp
// Using tiling fields: M, N, K, tileM, tileN, tileK
for m, n, k tiles:
    CopyIn: A[m,k], B[k,n]
    Compute: C += Mmad(A, B)
    CopyOut: C[m,n]
```

## Tiling Fields Required
- M, N, K: uint32_t // matrix dimensions
- tileM, tileN, tileK: uint32_t // tile sizes (aligned to 16)

## Useful References
- APIs: [Mmad, MatMul]
- Examples: [matmul_custom]
</response>

### Ex4: Reject wrong paradigm
Tiling: generate, Paradigm: vector (wrong for matmul!)
<response>
## Reasoning
MatMul proposed with vector paradigm, but matmul requires cube.

accepted: false

## Issues
1. Paradigm mismatch: matmul requires cube, not vector

## Suggestions
Change paradigm to cube. Use tile sizes aligned to 16.
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

(If true:)
## Kernel Design
...

## Kernel Pseudocode
...

## Tiling Fields Required
...

## Useful References
...

(If false:)
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

---

## Output Format

<response>
## Reasoning
<Any assumptions or workarounds needed? If none, write "Proposal is correct.">

accepted: true

## Kernel Design
- Paradigm: <vector | cube>
- Pipeline: <double_buffer>
- Operations: [op1, op2, ...]

## Kernel Pseudocode
```cpp
// Using tiling fields: <fields>
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

## Tiling Fields Required
- <field>: <type> // <purpose>

## Useful References
- APIs: [...]
- Examples: [...]
</response>

This design will be used for code generation. Make it complete!
"""
