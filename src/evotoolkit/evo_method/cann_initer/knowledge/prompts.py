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

1. **For API requests**:
   - Check if the API name exists in "Available APIs"
   - If exact match, keep it
   - If not found, find the closest API or mark as "skip" (with reason)

2. **For Example requests**:
   - Map conceptual names to actual operator names in "Available Operator Examples"
   - "attention operators" → find specific like "flash_attention_score"
   - "bmm (batch matmul)" → find "batch_matmul" or similar
   - If no match, mark as "skip" (with reason)

3. **You may**:
   - Remove duplicates
   - Add essential APIs/Examples that are clearly needed but missing
   - Prioritize based on relevance to the design

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
- [TYPE] ORIGINAL: REASON

## Analysis
Your brief explanation of the decisions made.
</retrieval_plan>

**Format rules**:
- PRIORITY: high, medium, or low
- TYPE: api or example
- Each item on its own line starting with "- "
- If a section is empty, write "None"

**Example output**:

<retrieval_plan>
## API Requests
- MatMul [high]: Exact match, needed for matrix multiply in attention score
- ReduceMax [high]: Exact match, needed for online softmax numerator
- ReduceSum [high]: Exact match, needed for softmax denominator
- Exp [high]: Exact match, needed for softmax exponential
- Sub [medium]: Exact match, needed for numerical stability

## Example Requests
- flash_attention_score [high]: Best match for "attention operators", shows tiled attention pattern
- softmax [medium]: Direct match for "softmax implementation"

## Skipped
- [api] SetZero: Not found in available APIs, use Duplicate with 0 instead
- [example] online softmax: Concept covered by flash_attention_score

## Analysis
Mapped all core APIs for FlashAttention computation. Selected flash_attention_score as primary example since it demonstrates the online softmax pattern. Skipped SetZero as it's not a standard API.
</retrieval_plan>
"""


# =============================================================================
# KnowledgeSummarizer Prompt
# =============================================================================

SUMMARIZER_PROMPT = """## Role
你是 Ascend C 代码知识专家。根据当前任务，从检索到的算子示例中提取最相关的信息。

## 当前任务

**算子描述**: {operator_description}

**Kernel 伪代码**:
```
{kernel_pseudocode}
```

## 检索到的算子示例

{examples_content}

## 任务

从上述示例中选择最相关的 {max_examples} 个，对于选中的示例，提取：
1. 与当前任务相关的关键技术
2. Kernel 代码中最相关的片段（核心计算逻辑）
3. Tiling 代码中最相关的片段（tiling 计算逻辑）

## 输出格式

<example_summaries>
### example_name_1
**相关度**: high/medium
**理由**: 为什么这个示例对当前任务有参考价值

**关键技术**:
- 技术点1（如：分块 tiling 策略）
- 技术点2（如：流水线优化）

**Kernel 参考代码**:
```cpp
// 最相关的 kernel 代码片段
// 保留 Process/Compute 等核心函数
```

**Tiling 参考代码**:
```cpp
// 最相关的 tiling 代码片段
// 保留 TilingFunc 核心逻辑
```

### example_name_2
...
</example_summaries>

**精简规则**:
- Kernel: 只保留 Init/Process/Compute/CopyIn/CopyOut 等核心函数，删除 #include、namespace 等
- Tiling: 只保留 TilingFunc 或 tiling 计算函数，删除注册宏等
- 只选择真正相关的示例，不相关的不要输出
- 关键技术要具体，如 "分块处理" 比 "优化" 更有价值
"""
