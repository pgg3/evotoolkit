# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Knowledge Module Prompts

- RetrievalPlanner: 将 Phase 1 的概念性检索请求转换为精确检索请求
- KnowledgeSummarizer: 从原始知识中提取与任务相关的摘要
"""

# =============================================================================
# RetrievalPlanner Prompt
# =============================================================================

RETRIEVAL_PLANNER_PROMPT = """## Role
You are the **Retrieval Planner** in a multi-agent Ascend C code generation pipeline.

Your task: Convert conceptual knowledge requests into precise retrieval requests.

## Context

**Operator Description**: {operator_description}

**Kernel Pseudocode** (from design phase):
```
{kernel_pseudocode}
```

**Tiling Execution** (from design phase):
```
{tiling_execution}
```

**Tiling Fields**:
{tiling_fields}

## Available Knowledge

{available_knowledge}

## Raw Requests from Phase 1

{raw_requests}

## Instructions

### 1. For API requests
- Check if the API name exists in the API list above
- If exact match found, keep it
- If not found but a similar API exists, map to the correct name (APIs may have aliases or naming variants)
- If no reasonable match, mark as "skip" with reason

### 2. For Example requests
- Map conceptual names to actual operator names in the example list
- Use semantic matching: "attention operators" → "flash_attention_score", "bmm" → "batch_matmul"
- If no match, mark as "skip" with reason

### 3. You may
- Remove duplicates
- Add essential APIs that are clearly implied by the pseudocode but missing from requests
- Infer required APIs from operations in pseudocode (e.g., softmax needs Exp, ReduceSum, Div)

### 4. Priority rules
- **high**: Directly used in kernel pseudocode core computation
- **medium**: Supporting operations (data movement, synchronization)
- **low**: Optional or alternative implementations

## Output Format

Use the following structured format (NOT JSON):

<retrieval_plan>
## API Requests
- NAME [PRIORITY]: REASON
- NAME [PRIORITY]: REASON

## Example Requests
- NAME [PRIORITY]: REASON
- NAME [PRIORITY]: REASON

## Skipped
- [TYPE] "ORIGINAL_REQUEST": REASON

## Analysis
Brief explanation of key decisions.
</retrieval_plan>

**Format rules**:
- PRIORITY: high, medium, or low
- TYPE: api or example
- ORIGINAL_REQUEST: the original conceptual name from Phase 1
- Each item on its own line starting with "- "
- If a section is empty, write "None"

**Example output**:

<retrieval_plan>
## API Requests
- Mmad [high]: Core matrix multiply for Q*K^T and score*V
- Exp [high]: Softmax exponential computation
- ReduceSum [high]: Softmax denominator
- Sub [medium]: Numerical stability (x - max)
- DataCopy [medium]: Data movement between GM and UB

## Example Requests
- flash_attention_score [high]: Best match for "attention operators"
- softmax_custom [medium]: Reference for softmax pattern

## Skipped
- [api] "SetZero": Not a standard API, use Duplicate with scalar 0
- [example] "online softmax": Covered by flash_attention_score example

## Analysis
Mapped core computation APIs from pseudocode. Added DataCopy for data movement implied by tiling. Selected flash_attention_score as primary example for tiled attention pattern.
</retrieval_plan>
"""


# =============================================================================
# KnowledgeSummarizer Prompt
# =============================================================================

SUMMARIZER_PROMPT = """## Role
You are an Ascend C code knowledge expert. Your task is to **select** the most relevant examples and establish mappings to the current task.

## Current Task

**Operator Description**: {operator_description}

**Kernel Pseudocode**:
```
{kernel_pseudocode}
```

**Tiling Execution Pseudocode**:
```
{tiling_execution}
```

**Tiling Fields**:
{tiling_fields}

## Retrieved Operator Examples

> Note: The following code has been preprocessed to retain only core functions (Process/Compute/TilingFunc, etc.).
> Your task is to **select** the most relevant examples and establish mappings, NOT to further simplify the code.

{examples_content}

## Task

Select the {max_examples} most relevant examples from above. For each selected example:

1. **Establish Mapping**: Clearly map concepts/variables/patterns from the example to the current task
2. **Extract Patterns**: Summarize reusable implementation patterns (data flow, pipeline, API call sequence)
3. **Mark Not Applicable**: Point out parts of the example that don't apply to the current task (if any)
4. **Preserve Code**: Keep the most relevant Kernel and Tiling code snippets

## Output Format

<example_summaries>
### example_name_1
**Selection Reason**: Why this example is valuable for the current task

**Mapping to Current Task**:
- Example's `varA` → Current task's `varB`
- Example's XX computation pattern → Current task's YY implementation

**Implementation Patterns**:
- Data flow: GM → UB (DataCopy) → Compute → UB → GM
- Pipeline: double buffer / single buffer
- API sequence: DataCopy → Add/Mul/... → DataCopy

**Key Techniques**:
- Technique 1 (specific description, e.g., using ReduceMax + Sub for numerical stability)
- Technique 2

**Not Applicable** (if any):
- Example's XX doesn't apply because current task is YY

**Kernel Reference Code**:
```cpp
// Code most relevant to current task
```

**Tiling Reference Code**:
```cpp
// Code most relevant to current task
```

### example_name_2
...
</example_summaries>

## Selection Criteria

- **Prefer**: Examples with similar computation patterns or using the same APIs as the current task
- **Mapping must be specific**: Clearly map variables/parameters/patterns, not vague descriptions
- **Patterns must be reusable**: Extracted patterns should directly guide current task implementation
- **Not applicable must be clear**: Help downstream avoid incorrectly copying unsuitable code
"""
