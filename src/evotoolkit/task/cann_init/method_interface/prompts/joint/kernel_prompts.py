# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Kernel Agent Prompts for Joint Branch

This module contains all prompts for the Kernel Specialist agent:
- First round review prompt (with examples)
- Re-review prompt for revision rounds (concise)
- Final round prompt (must produce implementable design)
"""

from typing import List

from evotoolkit.task.cann_init.method_interface.prompts.phase0 import _format_signature
from .chip_specs import DEFAULT_CHIP, format_chip_spec
from .utils import extract_current_plan


class KernelPromptsMixin:
    """Prompts for the Kernel Specialist agent"""

    def get_kernel_review_prompt(
        self, context: dict, conversation: List[dict], is_final_round: bool = False
    ) -> str:
        """Generate prompt for Kernel Specialist to review the proposal"""
        formatted_sig = _format_signature(context.get('signature'))
        compute_pattern = context.get('compute_pattern', 'other')

        # Get hardware specification
        npu_type = context.get('npu_type', DEFAULT_CHIP)
        hw_spec = format_chip_spec(npu_type)

        # Extract current tiling plan and check if revision round
        current_plan = extract_current_plan(conversation)
        previous_feedback = None

        # Check if this is a revision round (kernel already gave feedback before)
        for msg in conversation:
            if msg.get('role') == 'kernel':
                previous_feedback = msg.get('content', '')
                break

        # Final round: must produce implementable design regardless of issues
        if is_final_round:
            return self._get_kernel_final_round_prompt(
                formatted_sig, compute_pattern, context.get('python_ref'),
                current_plan, previous_feedback, hw_spec
            )

        if previous_feedback and current_plan:
            return self._get_kernel_re_review_prompt(
                formatted_sig, compute_pattern, context.get('python_ref'),
                current_plan, previous_feedback
            )

        # First round: full prompt with examples
        return self._get_kernel_first_round_prompt(
            formatted_sig, compute_pattern, context.get('python_ref'),
            current_plan, hw_spec
        )

    def _get_kernel_first_round_prompt(
        self,
        formatted_sig: str,
        compute_pattern: str,
        python_ref: str,
        current_plan: str,
        hw_spec: str,
    ) -> str:
        """Generate full prompt for first round review (with examples)."""
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
{python_ref}
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
for (...) {{
    // CopyIn
    <load data using tiling fields>

    // Compute
    <operations using tiling fields>

    // CopyOut
    <store data using tiling fields>
}}
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
for (int i = 0; i < tileNum; i++) {{
    // CopyIn
    xLocal = LoadTile(xGm, i * tileLength, tileLength);
    yLocal = LoadTile(yGm, i * tileLength, tileLength);

    // Compute
    zLocal = Add(xLocal, yLocal, tileLength);

    // CopyOut
    StoreTile(zGm, i * tileLength, zLocal, tileLength);
}}
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
for (int row = 0; row < rowsPerCore; row++) {{
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
}}
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

    def _get_kernel_final_round_prompt(
        self,
        formatted_sig: str,
        compute_pattern: str,
        python_ref: str,
        current_plan: str,
        previous_feedback: str,
        hw_spec: str,
    ) -> str:
        """Generate prompt for final round - MUST produce implementable design.

        This is the last round of discussion. The kernel agent must accept
        the proposal (possibly with modifications) and ensure the combined
        tiling + kernel design is implementable, because the next phase
        will generate actual code based on this design.
        """
        feedback_section = ""
        if previous_feedback:
            feedback_section = f"""## Your Previous Feedback
{previous_feedback}

"""
        return f"""## Role
You are the **kernel agent**. This is the **FINAL ROUND** of discussion.

## CRITICAL: You MUST produce an implementable design now.

The next phase will generate actual code based on your output. You cannot reject - you must either:
1. Accept the current proposal if it's workable, OR
2. Accept with modifications - fix remaining issues yourself and output a complete design

**Do NOT output "accepted: false"** - that would leave us without a design for code generation.

## Hardware
{hw_spec}

## Context
- Pattern: `{compute_pattern}`

## Operator Signature
{formatted_sig}

## Python Reference
```python
{python_ref}
```

## Current Tiling Proposal
{current_plan}

{feedback_section}## Your Task

Even if the tiling proposal has minor issues, you must:
1. Accept it and work around the issues in your kernel design, OR
2. Make reasonable assumptions and document them

Output a **complete, implementable design** that will be used for code generation.

## Output Format (MUST use this format)
<response>
accepted: true

## Final Notes
<If you made any assumptions or workarounds due to tiling issues, document them here. Otherwise write "None">

## Kernel Design
- Pipeline: <single_buffer | double_buffer>
- Operations: [<conceptual op1>, <conceptual op2>, ...]

## Kernel Pseudocode
```cpp
// Using tiling fields: <field1>, <field2>, ...
// (If tiling fields are missing, define what you need and note it above)
for (...) {{
    // CopyIn
    <load data using tiling fields>

    // Compute
    <operations using tiling fields>

    // CopyOut
    <store data using tiling fields>
}}
```

## Tiling Fields Required
<List all tiling fields needed by the kernel, with types and purposes>
- <field>: <type> // <purpose>

## Useful References
- <name>: <why>
</response>

Remember: This design will be used directly for code generation. Make it complete and implementable!
"""
