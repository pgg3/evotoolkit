#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
知识检索模块测试 - RetrievalPlanner with LLM

测试 FlashAttention 案例：
- 将 Phase 1 的概念性请求转换为精确请求
- 验证 LLM 能否正确映射模糊名称

用法:
    python 6_knowledge_retrieval.py [--build-index]
"""

import json
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


def test_retrieval_planner():
    """测试 RetrievalPlanner - FlashAttention 案例"""
    print("=" * 60)
    print("Testing RetrievalPlanner - FlashAttention")
    print("=" * 60)

    # 1. 初始化 LLM
    print("\n[1] Initializing LLM...")
    llm = get_llm()

    def llm_call(prompt: str) -> str:
        response, _ = llm.get_response(prompt)
        return response

    # 2. 初始化知识库
    print("[2] Loading Knowledge Base...")
    config = KnowledgeBaseConfig()
    kb = RealKnowledgeBase(config)
    print(f"    Loaded: {len(kb.index['operators'])} operators, {len(kb.index['apis'])} APIs")

    # 3. 创建 RetrievalPlanner
    planner = RetrievalPlanner(kb, llm_client=llm_call)

    # =========================================================================
    # FlashAttention 测试案例
    # =========================================================================

    operator_description = """
FlashAttention: Memory-efficient attention mechanism.
Input: Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D]
Output: O[B,H,S,D]
Computation: O = softmax(Q @ K^T / sqrt(D)) @ V
"""

    kernel_pseudocode = """
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
"""

    tiling_execution = """
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
"""

    tiling_fields = [
        {"name": "batchSize", "type": "uint32_t", "purpose": "batch dimension B"},
        {"name": "numHeads", "type": "uint32_t", "purpose": "number of attention heads H"},
        {"name": "seqLen", "type": "uint32_t", "purpose": "sequence length S"},
        {"name": "headDim", "type": "uint32_t", "purpose": "head dimension D"},
        {"name": "blockSize", "type": "uint32_t", "purpose": "tile size for sequence dimension"},
        {"name": "scale", "type": "float", "purpose": "1/sqrt(headDim)"},
    ]

    # 模拟 Phase 1 的概念性请求
    raw_requests = [
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
        {"type": "example", "name": "online softmax"},
        {"type": "example", "name": "batch matmul"},
    ]

    # =========================================================================
    # 显示输入
    # =========================================================================

    print("\n[3] Input - Operator Description:")
    print("-" * 40)
    print(operator_description.strip())

    print("\n[4] Input - Tiling Fields:")
    print("-" * 40)
    for f in tiling_fields:
        print(f"  - {f['name']}: {f['type']} // {f['purpose']}")

    print("\n[5] Input - Raw Requests from Phase 1:")
    print("-" * 40)
    for req in raw_requests:
        print(f"  [{req['type']}] {req['name']}")

    # =========================================================================
    # 调用 RetrievalPlanner
    # =========================================================================

    print("\n[6] Calling RetrievalPlanner with LLM...")
    print("-" * 40)

    result = planner.plan(
        operator_description=operator_description,
        kernel_pseudocode=kernel_pseudocode,
        tiling_execution=tiling_execution,
        tiling_fields=tiling_fields,
        raw_requests=raw_requests,
    )

    # =========================================================================
    # 显示结果
    # =========================================================================

    print("\n[7] Result - API Requests:")
    print("-" * 40)
    for req in result.get("api_requests", []):
        priority = req.get("priority", "?")
        name = req.get("name", "?")
        reason = req.get("reason", "")
        print(f"  [{priority}] {name}")
        if reason:
            print(f"       -> {reason}")

    print("\n[8] Result - Example Requests:")
    print("-" * 40)
    for req in result.get("example_requests", []):
        priority = req.get("priority", "?")
        name = req.get("name", "?")
        reason = req.get("reason", "")
        print(f"  [{priority}] {name}")
        if reason:
            print(f"       -> {reason}")

    print("\n[9] Result - Skipped:")
    print("-" * 40)
    for skip in result.get("skipped", []):
        print(f"  [{skip.get('type', '?')}] {skip.get('original', '?')}")
        print(f"       -> {skip.get('reason', '')}")

    print(f"\n[10] Analysis:")
    print("-" * 40)
    print(result.get("analysis", "N/A"))

    print("\n[11] Full JSON Result:")
    print("-" * 40)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Done!")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Retrieval Test")
    parser.add_argument("--build-index", action="store_true", help="Build knowledge index first")
    args = parser.parse_args()

    if args.build_index:
        print("Building knowledge index...")
        config = KnowledgeBaseConfig()
        builder = KnowledgeIndexBuilder(config)
        builder.build_index()
        print("Done!\n")

    test_retrieval_planner()


if __name__ == "__main__":
    main()
