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
        return f"""## 1. Role

You are the **Kernel Agent** in a multi-agent Ascend C code generation pipeline.

**Your responsibilities**:
1. **Review** the Tiling Agent's proposal for correctness
2. **Design** the kernel implementation strategy (if accepted)

This is the conceptual design phase. You will NOT write actual code.

---

## 2. Background

### Ascend C Architecture

| Component | Role |
|-----------|------|
| **Host (CPU)** | Computes tiling parameters, launches kernel |
| **Kernel (NPU)** | Executes computation on tiles |
| **TilingData** | Struct that passes parameters from Host to Kernel |

### Kernel Design Elements

| Element | Description |
|---------|-------------|
| **Pipeline** | Always use `double_buffer` (overlap compute with data transfer) |
| **Pseudocode** | High-level kernel logic (CopyIn → Compute → CopyOut) |
| **Tiling Fields** | Parameters defined in TilingData, directly accessible in kernel |

### About Useful References

At the end of your output, list APIs/examples for **knowledge retrieval**:
- **APIs**: Include both kernel APIs (compute operations) and tiling APIs (data movement)
- **Examples**: Similar operators to reference (e.g., "attention" → flash_attention)

Note: Conceptual names OK. A downstream planner will map to actual KB entries.

---

## 3. Your Task

Review this tiling proposal and design the kernel strategy.

### Operator Signature

{formatted_sig}

### Python Reference

```python
{python_ref}
```

### Hardware

{hw_spec}

### Tiling Proposal (from Tiling Agent)

{current_plan}

---

## 4. Review Flow

### Step 1: Review Tiling Proposal

Check each item:
- [ ] **Memory**: Tile fits in UB? (see Memory Check)
- [ ] **Attrs**: All scalar parameters included in Tiling Fields?
- [ ] **Multi-core**: No per-core values in Tiling Fields? (computed via `GetBlockIdx()`)
- [ ] **Block dim**: ≤ core count?

### Step 2: Design Kernel (if accept)

1. Write Kernel Pseudocode (CopyIn → Compute → CopyOut)
2. Confirm/adjust Tiling Fields
3. List Useful References (APIs and Examples for knowledge retrieval)

---

## 5. Guidelines

### Common Mistakes

- ✗ Requiring `blockOffset` in Tiling Fields → each core computes via `GetBlockIdx()`
- ✗ Forgetting to list required Tiling Fields → kernel needs these to compile

### Edge Case

- If data doesn't divide evenly, last tile handles remainder (kernel uses `min(tileLength, remaining)`)

---

## 6. Output Format

<response>
## Reasoning
<1-2 sentences: Is proposal correct? What's the kernel strategy?>

accepted: <true | false>
strategy: <default | generate>

(If accepted=true:)

## Kernel Design
- Pipeline: double_buffer

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
for (...) {{
    CopyIn: <load data>
    Compute: <operations>
    CopyOut: <store result>
}}
```

## Useful References
- APIs: [<API names for implementation>]
- Examples: [<similar operators>]

(If strategy=generate:)

## Tiling Fields Alignment
- Match: [<fields matching proposal>]
- Missing: [<fields needed but not in proposal>] (or "none")
- Unused: [<fields in proposal but not needed>] (or "none")

## Tiling Fields Required
- <field>: <type> // <purpose>

(If accepted=false:)

## Issues
1. <what's wrong>

## Suggestions
<how to fix>
</response>

---

## 7. Examples

### Ex1: Accept ReLU (default)
Tiling proposal: Strategy: default (single input, same shape)

<response>
## Reasoning
ReLU is single-input element-wise. Default template is correct.

accepted: true
strategy: default

## Kernel Design
- Pipeline: double_buffer

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
for (int i = 0; i < tileNum; i++) {{
    CopyIn: x[offset : offset + tileLength]
    Compute: y = Relu(x)
    CopyOut: y[offset : offset + tileLength]
}}
```

## Useful References
- APIs: [Relu, DataCopy]
- Examples: [relu_custom]
</response>

### Ex2: Accept Add (generate)
Tiling proposal: Strategy: generate, Tiling Fields: [totalLength, tileNum, tileLength]

<response>
## Reasoning
Add has two inputs, needs generate. Proposal is correct.

accepted: true
strategy: generate

## Kernel Design
- Pipeline: double_buffer

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
for (int i = 0; i < tileNum; i++) {{
    CopyIn: x[idx], y[idx]
    Compute: z = Add(x, y)
    CopyOut: z[idx]
}}
```

## Useful References
- APIs: [Add, DataCopy]
- Examples: [add_custom]

## Tiling Fields Alignment
- Match: [totalLength, tileNum, tileLength]
- Missing: none
- Unused: none

## Tiling Fields Required
- totalLength: uint32_t // total elements
- tileNum: uint32_t // tiles per core
- tileLength: uint32_t // elements per tile
</response>

### Ex3: Accept Softmax (generate - reduction)
Tiling proposal: Strategy: generate, row-wise reduction

<response>
## Reasoning
Softmax needs row-wise reduction. Proposal is correct.

accepted: true
strategy: generate

## Kernel Design
- Pipeline: double_buffer

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for row offset
for (int row = 0; row < rowsPerCore; row++) {{
    CopyIn: x[row, :]
    Compute: max = ReduceMax(x)
             x = Sub(x, max)
             x = Exp(x)
             sum = ReduceSum(x)
             y = Div(x, sum)
    CopyOut: y[row, :]
}}
```

## Useful References
- APIs: [ReduceMax, ReduceSum, Exp, Sub, Div, DataCopy]
- Examples: [softmax_custom]

## Tiling Fields Alignment
- Match: [batchSize, featureDim, rowsPerCore]
- Missing: none
- Unused: none

## Tiling Fields Required
- batchSize: uint32_t // total rows
- featureDim: uint32_t // row length
- rowsPerCore: uint32_t // rows per core
</response>

### Ex4: Reject - Missing Attrs
Tiling proposal: Strategy: generate for Clamp(x, min_val, max_val), but Tiling Fields only has [totalLength, tileNum, tileLength]

<response>
## Reasoning
Clamp has Attrs (min_val, max_val) but they're missing from Tiling Fields. Kernel cannot compile without these values.

accepted: false
strategy: generate

## Issues
1. Missing Attrs: min_val and max_val not in Tiling Fields

## Suggestions
Add to Tiling Fields:
- minVal: float // clamp lower bound
- maxVal: float // clamp upper bound
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

---

## Context

**Operator**: {formatted_sig}

```python
{python_ref}
```

---

## Revised Tiling Proposal

{current_plan}

---

## Your Previous Feedback

{previous_feedback}

---

## Task

Check if the revision addresses your feedback. Output format:

<response>
## Reasoning
<Does revision address your feedback? Any remaining issues?>

accepted: <true | false>
strategy: <default | generate>

(If accepted=true:)

## Kernel Design
- Pipeline: double_buffer

## Kernel Pseudocode
```cpp
// Multi-core: GetBlockIdx() for core offset
...
```

## Useful References
- APIs: [...]
- Examples: [...]

(If strategy=generate:)

## Tiling Fields Alignment
- Match: [...]
- Missing: [...] (or "none")
- Unused: [...] (or "none")

## Tiling Fields Required
- <field>: <type> // <purpose>

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

---

"""
        return f"""## Role

You are the **Kernel Agent**. This is the **FINAL ROUND**.

---

## CRITICAL

You **MUST** produce an implementable design now. You cannot reject.

Either:
1. Accept if proposal is workable, OR
2. Accept with modifications - fix issues yourself and note in `## Assumptions Made`

**Do NOT output `accepted: false`**

---

## Context

**Operator**: {formatted_sig}

```python
{python_ref}
```

**Hardware**: {hw_spec}

---

## Current Tiling Proposal

{current_plan}

---

{feedback_section}## Output Format

**IMPORTANT**: You MUST include `## Tiling Execution` section (this marks final round).

<response>
## Reasoning
<Brief assessment. If you made modifications, explain what and why.>

accepted: true
strategy: <default | generate>

(If you made modifications:)
## Assumptions Made
- <what you modified from original proposal>
- <e.g., "Added lastTileLength for remainder handling">

## Kernel Design
- Pipeline: double_buffer

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
    CopyIn: <data with index>
    CopyOut: <data with index>
```
Note: <why this pattern>

## Useful References
- APIs: [...]
- Examples: [...]

(If strategy=generate:)

## Tiling Fields Alignment
- Match: [...]
- Missing: [...] (or "none")
- Unused: [...] (or "none")

## Tiling Fields Required
- <field>: <type> // <purpose>
</response>

This design will be used for code generation. Make it complete!
"""
