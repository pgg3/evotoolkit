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

### Operator Signature

{operator_signature}

### Kernel Pseudocode

```
{kernel_pseudocode}
```

### Tiling Execution

```
{tiling_execution}
```

### Tiling Fields

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
- **IMPORTANT**: The following are kernel class METHOD NAMES, NOT system APIs to retrieve:
  - `Init`, `Process`, `Compute`, `Calculate` - kernel class lifecycle methods
  - `CopyIn`, `CopyOut`, `CopyInAndCast`, `CastAndCopyOut` - kernel class data movement methods
  - These are defined by the user in the kernel class, not provided by the Ascend C SDK
  - Skip these with reason "Kernel class method, not a system API"

### 2. For Example requests
- Map conceptual names to actual operator names in the example list
- Use semantic matching: "attention operators" → "flash_attention_score", "bmm" → "batch_matmul"
- If no match, mark as "skip" with reason

**IMPORTANT: Smart Example Selection based on Operator Complexity**

Analyze the kernel pseudocode to determine operator complexity, then select the appropriate number of examples:

| Complexity | Characteristics | Examples Needed |
|------------|-----------------|-----------------|
| **Simple** | Single element-wise op (Relu, Abs, Exp), no reduce, default tiling | **1 example** (most similar one) |
| **Medium** | Reduce ops (Softmax, LayerNorm), or 2-3 element-wise ops combined | **1-2 examples** |
| **Complex** | MatMul, Attention, multi-stage computation, custom tiling | **2-3 examples** |

Selection criteria:
- Prefer examples with **same operation type** (element-wise → element-wise example)
- Prefer examples with **similar data flow** (single input/output → similar example)
- Avoid redundant examples (don't pick two examples that show the same pattern)
- Skip examples that are much more complex than needed (don't use attention example for Relu)

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
- **Operator Complexity**: [simple/medium/complex] - [brief reason]
- **Example Selection**: Selected N example(s) because [reason]
- **Key Decisions**: [other important decisions]
</retrieval_plan>

**Format rules**:
- PRIORITY: high, medium, or low
- TYPE: api or example
- ORIGINAL_REQUEST: the original conceptual name from Phase 1
- Each item on its own line starting with "- "
- If a section is empty, write "None"

**Example output (Complex operator - Attention)**:

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
- **Operator Complexity**: complex - Multi-stage attention with MatMul + Softmax + MatMul
- **Example Selection**: Selected 2 examples: flash_attention for overall pattern, softmax for numerical stability
- **Key Decisions**: Added DataCopy for data movement implied by tiling
</retrieval_plan>

**Example output (Simple operator - Relu)**:

<retrieval_plan>
## API Requests
- Relu [high]: Core element-wise activation
- DataCopy [medium]: Data movement between GM and UB

## Example Requests
- squared_relu [high]: Most similar element-wise pattern with Relu API usage

## Skipped
- [example] "gelu_mul": Too complex for simple Relu (has Gelu + Mul, we only need Relu)
- [example] "elementwise_unary": Redundant, squared_relu already covers the pattern

## Analysis
- **Operator Complexity**: simple - Single element-wise Relu operation with default tiling
- **Example Selection**: Selected 1 example (squared_relu) - directly uses Relu API with same data flow
- **Key Decisions**: Skipped more complex examples to avoid unnecessary information
</retrieval_plan>
"""


# =============================================================================
# KnowledgeSummarizer Prompt
# =============================================================================

SUMMARIZER_PROMPT = """## Role
You are an Ascend C code expert. Extract **reusable code patterns** from reference examples for implementing the current task.

## Current Task

### Operator Signature

{operator_signature}

### Kernel Pseudocode

```
{kernel_pseudocode}
```

### Tiling Execution

```
{tiling_execution}
```

### Tiling Fields

{tiling_fields}

## Retrieved Operator Examples

{examples_content}

## Task

Select the {max_examples} most relevant examples. For each, extract **complete, reusable code patterns** (not summaries).

## Output Format

<example_summaries>
### example_name

**Why Selected**: One sentence explaining relevance to current task.

**Adaptation Notes**: What to change when adapting to current task (variable names, API calls, etc.)

**Init Pattern** (buffer setup):
```cpp
// COMPLETE Init function - copy the actual code
// Include: GlobalTensor setup, LocalTensor declarations, pipe.InitBuffer
```

**Process Pattern** (main loop):
```cpp
// COMPLETE Process function - copy the actual code
// Include: loop structure, offset calculation, tile iteration
```

**Compute Pattern** (core computation):
```cpp
// COMPLETE Compute function - copy the actual code
// Include: API calls like Relu/Add/Mul, PipeBarrier, data type handling
```

**CopyIn/CopyOut Pattern** (data movement):
```cpp
// DataCopy/DataCopyPad calls with actual parameters
// Include: DataCopyExtParams setup, offset calculation
```

**Key API Calls** (critical for implementation):
```cpp
// Extract the EXACT API call patterns used, one per line:
// Relu(dst, src, count);
// DataCopy(dst, src, count);
// PipeBarrier<PIPE_V>();
```

</example_summaries>

## Requirements

1. **Copy actual code** - Do NOT summarize or simplify. Keep complete functions.
2. **Include all API calls** - Every DataCopy, Relu, Add, Mul, Cast, PipeBarrier, SetFlag/WaitFlag.
3. **Keep parameter details** - The exact parameters are critical for correct implementation.
4. **Preserve template patterns** - If code uses templates, keep the template structure.
5. **Key API Calls section is CRITICAL** - This helps the implementation agent use correct API syntax.
"""
