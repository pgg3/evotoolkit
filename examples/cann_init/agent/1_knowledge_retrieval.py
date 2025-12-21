#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
知识检索模块独立测试

测试 FlashAttention 案例的完整知识检索流程:
1. RetrievalPlanner: 概念性请求 → 精确请求
2. KnowledgeBase: 精确请求 → 原始知识
3. KnowledgeSummarizer: 原始知识 → 精简摘要

用法:
    python 1_knowledge_retrieval.py [--build-index]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import get_llm, get_knowledge_base

from evotoolkit.evo_method.cann_initer import (
    KnowledgeIndexBuilder,
    RetrievalPlanner,
    KnowledgeSummarizer,
)


# =============================================================================
# Test Context: FlashAttention (simulated joint plan output)
# =============================================================================

FLASH_ATTENTION_CONTEXT = {
    # operator_signature: 使用与 kernel_prompts.py 一致的格式
    "operator_signature": {
        "op_name": "FlashAttention",
        "inputs": [
            {"name": "q", "dtype": "float16", "is_tensor": True},
            {"name": "k", "dtype": "float16", "is_tensor": True},
            {"name": "v", "dtype": "float16", "is_tensor": True},
        ],
        "outputs": [
            {"name": "o", "dtype": "float16", "is_tensor": True},
        ],
        "init_params": [],
    },
    "kernel_pseudocode": """
// FlashAttention kernel pseudocode
for each batch b:
    for each query block q_block:
        load Q_block[q_block]
        init O_block = 0, l_block = 0, m_block = -inf

        for each key block k_block:
            load K_block[k_block], V_block[k_block]

            // Compute attention scores
            S = Q_block @ K_block^T  // MatMul
            m_new = max(m_block, rowmax(S))
            P = exp(S - m_new)  // Softmax numerator
            l_new = exp(m_block - m_new) * l_block + rowsum(P)

            // Update output
            O_block = exp(m_block - m_new) * O_block + P @ V_block

            m_block = m_new
            l_block = l_new

        O_block = O_block / l_block  // Normalize
        store O_block
""",
    "tiling_execution": """
block_dim = batch_size (parallelize over batches)
tile_num = num_query_blocks * num_kv_blocks

for each tile:
    CopyIn: Q_block, K_block, V_block
    Compute: MatMul(Q,K^T), Softmax, MatMul(P,V)
    CopyOut: O_block (accumulated)
""",
    "tiling_fields": [
        {"name": "batchSize", "type": "uint32_t", "purpose": "batch dimension"},
        {"name": "seqLen", "type": "uint32_t", "purpose": "sequence length"},
        {"name": "headDim", "type": "uint32_t", "purpose": "head dimension"},
        {"name": "blockSize", "type": "uint32_t", "purpose": "tile block size"},
    ],
}

RAW_REQUESTS = [
    {"type": "api", "name": "MatMul"},
    {"type": "api", "name": "Softmax"},
    {"type": "api", "name": "DataCopy"},
    {"type": "example", "name": "flash_attention"},
    {"type": "example", "name": "matmul"},
]


# =============================================================================
# Main
# =============================================================================

def main(build_index: bool = False):
    print("=" * 70)
    print("Full Knowledge Retrieval Pipeline Test - FlashAttention")
    print("=" * 70)

    # 1. Initialize
    print("\n[1] Initializing components...")
    llm = get_llm()

    def llm_call(prompt: str) -> str:
        response, _ = llm.get_response(prompt)
        return response

    kb = get_knowledge_base()
    print(f"    Knowledge Base: {kb.get_api_count()} APIs, {kb.get_operator_count()} operators")

    # Optional: Build index
    if build_index:
        print("\n[1.5] Building index...")
        builder = KnowledgeIndexBuilder(kb.config)
        builder.build_all()
        # Reload
        kb = get_knowledge_base()
        print(f"    After rebuild: {kb.get_api_count()} APIs, {kb.get_operator_count()} operators")

    # 2. Stage 1: RetrievalPlanner
    print("\n" + "=" * 70)
    print("[2] Stage 1: RetrievalPlanner")
    print("=" * 70)

    planner = RetrievalPlanner(kb, llm_client=llm_call)
    plan_result = planner.plan(
        operator_signature=FLASH_ATTENTION_CONTEXT["operator_signature"],
        kernel_pseudocode=FLASH_ATTENTION_CONTEXT["kernel_pseudocode"],
        tiling_execution=FLASH_ATTENTION_CONTEXT["tiling_execution"],
        tiling_fields=FLASH_ATTENTION_CONTEXT["tiling_fields"],
        raw_requests=RAW_REQUESTS,
    )

    print("\n--- API Requests ---")
    for req in plan_result.get("api_requests", []):
        print(f"  [{req.get('priority', '?')}] {req.get('name', '?')}")

    print("\n--- Example Requests ---")
    for req in plan_result.get("example_requests", []):
        print(f"  [{req.get('priority', '?')}] {req.get('name', '?')}")

    # 3. Stage 2: Fetch raw knowledge
    print("\n" + "=" * 70)
    print("[3] Stage 2: Fetch Raw Knowledge")
    print("=" * 70)

    raw_knowledge = {"apis": {}, "examples": {}}

    for req in plan_result.get("api_requests", []):
        name = req.get("name")
        if name:
            result = kb.search_api(name)
            raw_knowledge["apis"][name] = result
            print(f"  API '{name}': {len(result.get('content', ''))} chars")

    for req in plan_result.get("example_requests", []):
        name = req.get("name")
        if name:
            result = kb.search_operator(name)
            raw_knowledge["examples"][name] = result
            if result:
                print(f"  Example '{name}': {len(str(result))} chars")
            else:
                print(f"  Example '{name}': not found")

    # 4. Stage 3: KnowledgeSummarizer
    print("\n" + "=" * 70)
    print("[4] Stage 3: KnowledgeSummarizer")
    print("=" * 70)

    cann_path = getattr(kb.config, 'cann_path', None)
    summarizer = KnowledgeSummarizer(
        llm_client=llm_call,
        max_examples=2,
        cann_path=cann_path,
    )

    summarized = summarizer.summarize(
        task_context=FLASH_ATTENTION_CONTEXT,
        raw_knowledge=raw_knowledge,
    )

    print("\n--- API Summaries ---")
    for summary in summarized.get("api_summaries", []):
        print(f"  {summary.get('name')}: {len(summary.get('summary', ''))} chars")

    print("\n--- Example Summaries ---")
    for summary in summarized.get("example_summaries", []):
        print(f"  {summary.get('name')}: {len(summary.get('summary', ''))} chars")

    print("\n--- Combined Context (first 1000 chars) ---")
    combined = summarized.get("combined_context", "")
    print(combined[:1000] + "..." if len(combined) > 1000 else combined)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    build_index = "--build-index" in sys.argv
    main(build_index=build_index)
