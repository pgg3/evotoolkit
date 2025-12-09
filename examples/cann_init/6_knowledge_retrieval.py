#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
知识检索模块测试 - 完整 Pipeline 测试

测试 FlashAttention 案例的完整知识检索流程:
1. RetrievalPlanner: 概念性请求 → 精确请求
2. KnowledgeBase: 精确请求 → 原始知识
3. KnowledgeSummarizer: 原始知识 → 精简摘要

用法:
    python 6_knowledge_retrieval.py [--build-index]
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from evotoolkit.tools.llm import HttpsApi
from evotoolkit.evo_method.cann_initer import (
    RealKnowledgeBase,
    KnowledgeBaseConfig,
    KnowledgeIndexBuilder,
    RetrievalPlanner,
    KnowledgeSummarizer,
)

# Load .env from current directory
load_dotenv(Path(__file__).parent / ".env")


def get_llm():
    """Get LLM instance from environment variables."""
    api_url = os.getenv("API_URL")
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL", "gpt-4o")

    if not api_url or not api_key:
        raise ValueError(
            "API_URL and API_KEY must be set in .env file.\n"
            "Example .env:\n"
            "  API_URL=ai.api.xn--fiqs8s\n"
            "  API_KEY=sk-xxx\n"
            "  MODEL=claude-sonnet-4-5-20250929"
        )

    return HttpsApi(api_url=api_url, key=api_key, model=model)


# =============================================================================
# FlashAttention 测试数据
# =============================================================================

FLASH_ATTENTION_CONTEXT = {
    "operator_description": """
FlashAttention: Memory-efficient attention mechanism.
Input: Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D]
Output: O[B,H,S,D]
Computation: O = softmax(Q @ K^T / sqrt(D)) @ V
""",

    "kernel_pseudocode": """
// Using tiling fields: batchSize, numHeads, seqLen, headDim, blockSize
for (int b = 0; b < batchSize; b++) {
    for (int h = 0; h < numHeads; h++) {
        for (int i = 0; i < seqLen; i += blockSize) {
            // CopyIn: load Q block
            qLocal = LoadTile(qGm, offset_q, blockSize * headDim);

            // Initialize accumulators
            maxVal = -INF;
            sumVal = 0;
            oLocal = zeros(blockSize, headDim);

            for (int j = 0; j < seqLen; j += blockSize) {
                // CopyIn: load K, V blocks
                kLocal = LoadTile(kGm, offset_k, blockSize * headDim);
                vLocal = LoadTile(vGm, offset_v, blockSize * headDim);

                // Compute: attention score
                score = MatMul(qLocal, Transpose(kLocal));  // [blockSize, blockSize]
                score = Muls(score, scale);                 // scale = 1/sqrt(D)

                // Compute: online softmax
                newMax = ReduceMax(score);
                maxVal = Max(maxVal, newMax);
                score = Sub(score, maxVal);
                score = Exp(score);
                sumVal = sumVal * Exp(oldMax - maxVal) + ReduceSum(score);

                // Compute: weighted sum
                oLocal = oLocal * Exp(oldMax - maxVal) + MatMul(score, vLocal);
            }

            // Normalize
            oLocal = Div(oLocal, sumVal);

            // CopyOut
            StoreTile(oGm, offset_o, oLocal, blockSize * headDim);
        }
    }
}
""",

    "tiling_execution": """
for b in range(batchSize):
    for h in range(numHeads):
        for i_block in range(seqLen // blockSize):
            CopyIn: Q[b,h,i_block]
            init: maxVal=-INF, sumVal=0, O=zeros

            for j_block in range(seqLen // blockSize):
                CopyIn: K[b,h,j_block], V[b,h,j_block]
                Compute:
                    score = Q @ K.T * scale
                    online_softmax_update(score, maxVal, sumVal)
                    O += softmax(score) @ V

            CopyOut: O[b,h,i_block] / sumVal
""",

    "tiling_fields": [
        {"name": "batchSize", "type": "uint32_t", "purpose": "batch dimension B"},
        {"name": "numHeads", "type": "uint32_t", "purpose": "number of attention heads H"},
        {"name": "seqLen", "type": "uint32_t", "purpose": "sequence length S"},
        {"name": "headDim", "type": "uint32_t", "purpose": "head dimension D"},
        {"name": "blockSize", "type": "uint32_t", "purpose": "tile size for sequence dimension"},
        {"name": "scale", "type": "float", "purpose": "1/sqrt(headDim)"},
    ],
}

# 模拟 Phase 1 的概念性请求
RAW_REQUESTS = [
    # APIs - 有些精确，有些模糊
    {"type": "api", "name": "MatMul"},
    {"type": "api", "name": "Transpose"},
    {"type": "api", "name": "Muls"},
    {"type": "api", "name": "ReduceMax"},
    {"type": "api", "name": "ReduceSum"},
    {"type": "api", "name": "Sub"},
    {"type": "api", "name": "Exp"},
    {"type": "api", "name": "Div"},
    {"type": "api", "name": "Max"},
    {"type": "api", "name": "SetZero"},
    # Examples - 概念性名称
    {"type": "example", "name": "flash attention"},
    {"type": "example", "name": "attention operators"},
    {"type": "example", "name": "softmax implementation"},
]


def test_full_pipeline():
    """测试完整流程：RetrievalPlanner → KnowledgeBase → KnowledgeSummarizer"""
    print("=" * 70)
    print("Full Knowledge Retrieval Pipeline Test - FlashAttention")
    print("=" * 70)

    # 1. 初始化
    print("\n[1] Initializing components...")
    llm = get_llm()
    def llm_call(prompt: str) -> str:
        response, _ = llm.get_response(prompt)
        return response

    config = KnowledgeBaseConfig()
    kb = RealKnowledgeBase(config)
    print(f"    Knowledge Base: {kb.get_api_count()} APIs, {kb.get_operator_count()} operators")

    # 2. Stage 1: RetrievalPlanner
    print("\n" + "=" * 70)
    print("[2] Stage 1: RetrievalPlanner")
    print("=" * 70)

    planner = RetrievalPlanner(kb, llm_client=llm_call)
    plan_result = planner.plan(
        operator_description=FLASH_ATTENTION_CONTEXT["operator_description"],
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

    print("\n--- Skipped ---")
    for skip in plan_result.get("skipped", []):
        print(f"  [{skip.get('type', '?')}] {skip.get('original', '?')}: {skip.get('reason', '')}")

    # 3. Fetch raw knowledge
    print("\n" + "=" * 70)
    print("[3] Fetching Raw Knowledge")
    print("=" * 70)

    raw_knowledge = {"apis": {}, "examples": {}}

    for req in plan_result.get("api_requests", []):
        name = req.get("name")
        if name:
            raw_knowledge["apis"][name] = kb.search_api(name)
            status = raw_knowledge["apis"][name].get("status", "?")
            print(f"  API {name}: {status}")

    for req in plan_result.get("example_requests", []):
        name = req.get("name")
        if name:
            raw_knowledge["examples"][name] = kb.search_operator(name)
            confidence = raw_knowledge["examples"][name].get("confidence", "?")
            print(f"  Example {name}: {confidence}")

    # 4. Stage 2: KnowledgeSummarizer
    print("\n" + "=" * 70)
    print("[4] Stage 2: KnowledgeSummarizer")
    print("=" * 70)

    summarizer = KnowledgeSummarizer(
        llm_client=llm_call,
        max_examples=2,
        cann_path=config.cann_path,
    )

    summarized = summarizer.summarize(
        task_context={
            "operator_description": FLASH_ATTENTION_CONTEXT["operator_description"],
            "kernel_pseudocode": FLASH_ATTENTION_CONTEXT["kernel_pseudocode"],
            "tiling_execution": FLASH_ATTENTION_CONTEXT["tiling_execution"],
            "tiling_fields": FLASH_ATTENTION_CONTEXT["tiling_fields"],
        },
        raw_knowledge=raw_knowledge,
    )

    print(f"\n--- API Summaries ({len(summarized['api_summaries'])} APIs) ---")
    for api in summarized["api_summaries"]:
        print(f"  {api['name']}: {api.get('description', '')[:50]}...")

    print(f"\n--- Example Summaries ({len(summarized['example_summaries'])} examples) ---")
    for ex in summarized["example_summaries"]:
        print(f"  {ex['name']}: {ex.get('purpose', '')[:50]}...")

    # 5. Final combined context
    print("\n" + "=" * 70)
    print("[5] Combined Context (for Implementation Agent)")
    print("=" * 70)

    combined = summarized["combined_context"]
    print(f"\n{combined[:2000]}")
    if len(combined) > 2000:
        print(f"\n... (truncated, total {len(combined)} chars)")

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)

    return {
        "plan_result": plan_result,
        "raw_knowledge": raw_knowledge,
        "summarized": summarized,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Retrieval Pipeline Test")
    parser.add_argument("--build-index", action="store_true", help="Build knowledge index first")
    args = parser.parse_args()

    if args.build_index:
        print("Building knowledge index...")
        config = KnowledgeBaseConfig()
        builder = KnowledgeIndexBuilder(config)
        builder.build_index()
        print("Done!\n")

    test_full_pipeline()


if __name__ == "__main__":
    main()
