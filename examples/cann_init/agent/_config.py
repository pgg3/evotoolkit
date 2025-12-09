# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Agent 测试共享配置

提供:
1. 测试用例加载 (easy/medium/hard)
2. LLM 初始化
3. KnowledgeBase 初始化
4. 共享的上下文占位符 (运行后填入)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv(Path(__file__).parent / ".env")

# Directories
TEST_CASES_DIR = Path(__file__).parent / "test_cases"
OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = {
    "easy": {
        "name": "ReLU",
        "op_name": "Relu",
        "file": "easy_relu.py",
        "description": "Element-wise operation, uses default tiling",
        "max_debug_iterations": 3,
        "max_joint_turns": 2,
    },
    "medium": {
        "name": "Softmax",
        "op_name": "Softmax",
        "file": "medium_softmax.py",
        "description": "Reduce + element-wise, needs shape inference",
        "max_debug_iterations": 5,
        "max_joint_turns": 3,
    },
    "hard": {
        "name": "ScaledDotProductAttention",
        "op_name": "SDPA",
        "file": "hard_sdpa.py",
        "description": "MatMul + Softmax + MatMul, complex tiling",
        "max_debug_iterations": 8,
        "max_joint_turns": 5,
    },
}


def load_python_ref(test_case: str) -> str:
    """Load Python reference code from test case file."""
    config = TEST_CASES[test_case]
    file_path = TEST_CASES_DIR / config["file"]
    return file_path.read_text()


def get_test_config(test_case: str) -> dict:
    """Get test case configuration."""
    if test_case not in TEST_CASES:
        raise ValueError(f"Unknown test case: {test_case}. Available: {list(TEST_CASES.keys())}")
    return TEST_CASES[test_case]


# =============================================================================
# LLM & KnowledgeBase Initialization
# =============================================================================

def get_llm():
    """Get LLM instance from environment variables."""
    # Import here to avoid circular imports
    from evotoolkit.tools.llm import HttpsApi

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


def get_knowledge_base():
    """Get KnowledgeBase instance."""
    from evotoolkit.evo_method.cann_initer import RealKnowledgeBase, KnowledgeBaseConfig

    config = KnowledgeBaseConfig()
    return RealKnowledgeBase(config)


# =============================================================================
# Shared Context (TODO: Fill after running tests)
# =============================================================================

# Phase 0 output - fill after running 2_phase0.py
PHASE0_CONTEXT = {
    "easy": {
        "op_name": "Relu",
        "signature": None,  # TODO: Fill after running 2_phase0.py
        "compute_pattern": "element-wise",
        "strategies": {"tiling": "default", "pybind": "generate"},
    },
    "medium": {
        "op_name": "Softmax",
        "signature": None,  # TODO
        "compute_pattern": "reduction",
        "strategies": {"tiling": "custom", "pybind": "generate"},
    },
    "hard": {
        "op_name": "SDPA",
        "signature": {'op_name': 'SDPA', 'inputs': [{'name': 'q', 'dtype': 'float', 'is_tensor': True}, {'name': 'k', 'dtype': 'float', 'is_tensor': True}, {'name': 'v', 'dtype': 'float', 'is_tensor': True}], 'outputs': [{'name': 'output', 'dtype': 'float', 'is_tensor': True}], 'init_params': []},
        "compute_pattern": "other",
        "output_equals_input_shape": True,
        "shape_inference": {'input': 'Q=[B, S, D], K=[B, S, D], V=[B, S, D] where B=batch, S=sequence_length, D=embedding_dimension', 'output': '[B, S, D] (same as Q, K, V)', 'formula': 'auto output_shape = q.sizes();'},
        "functionality": 'Implements scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V, computing attention-weighted values where attention scores are derived from query-key similarity.',
        "strategies": {'kernel': 'generate', 'tiling': 'generate', 'pybind': 'generate'},
    },
}

# Joint Plan output - fill after running 4_joint_planning.py
JOINT_PLAN_CONTEXT = {
    "easy": None,    # TODO: Fill after running 4_joint_planning.py
    "medium": None,  # TODO
    "hard": {
        'tiling_strategy': 'custom',
        'tiling_fields': [
            {'name': 'batchSize', 'type': 'uint32_t', 'purpose': 'batch dimension (32)'},
            {'name': 'seqLen', 'type': 'uint32_t', 'purpose': 'sequence length dimension (128)'},
            {'name': 'dModel', 'type': 'uint32_t', 'purpose': 'model dimension (768)'},
            {'name': 'tileSeq', 'type': 'uint32_t', 'purpose': 'number of query rows per tile (4-8)'},
            {'name': 'scaleVal', 'type': 'float', 'purpose': '1/sqrt(dModel) for attention scaling'},
        ],
        'kernel_pseudocode': '''// Using tiling fields: batchSize, seqLen, dModel, tileSeq, scaleVal
for (int batch_idx = blockIdx / num_seq_tiles; batch_idx < batchSize; batch_idx += blockDim / num_seq_tiles) {
    for (int seq_tile_idx = blockIdx % num_seq_tiles; seq_tile_idx < seqLen / tileSeq; seq_tile_idx += step) {
        // CopyIn
        qLocal = LoadTile(qGm, qRowOffset, tileSeq * dModel);          // [tileSeq, dModel]
        kLocal = LoadTile(kGm, kOffset, seqLen * dModel);              // [seqLen, dModel]
        vLocal = LoadTile(vGm, vOffset, seqLen * dModel);              // [seqLen, dModel]

        // Compute: Q @ K^T
        scoresLocal = MatMul(qLocal, kLocal_transposed, tileSeq, seqLen, dModel);  // [tileSeq, seqLen]

        // Compute: Scale
        scoresLocal = Muls(scoresLocal, scaleVal, tileSeq * seqLen);

        // Compute: Softmax (row-wise over seqLen dimension)
        for (int row = 0; row < tileSeq; row++) {
            maxVal = ReduceMax(scoresLocal[row], seqLen);
            scoresLocal[row] = Sub(scoresLocal[row], maxVal, seqLen);
            scoresLocal[row] = Exp(scoresLocal[row], seqLen);
            sumVal = ReduceSum(scoresLocal[row], seqLen);
            scoresLocal[row] = Div(scoresLocal[row], sumVal, seqLen);
        }

        // Compute: scores @ V
        outLocal = MatMul(scoresLocal, vLocal, tileSeq, dModel, seqLen);  // [tileSeq, dModel]

        // CopyOut
        StoreTile(outGm, outOffset, outLocal, tileSeq * dModel);
    }
}''',
        'tiling_execution': '''for batch_idx in myBatches:
    for seq_tile in mySeqTiles:
        CopyIn: Q[batch_idx, seq_tile, :], K[batch_idx, :, :], V[batch_idx, :, :]
        Compute:
            - Cube: scores = Q_tile @ K^T
            - Vector: scores /= sqrt(d_model)
            - Vector: softmax(scores, dim=-1)
            - Cube: out = scores @ V
        CopyOut: output[batch_idx, seq_tile, :]''',
        'retrieval_requests': [
            {'type': 'api', 'name': 'MatMul'},
            {'type': 'api', 'name': 'Muls'},
            {'type': 'api', 'name': 'ReduceMax'},
            {'type': 'api', 'name': 'ReduceSum'},
            {'type': 'api', 'name': 'Sub'},
            {'type': 'api', 'name': 'Exp'},
            {'type': 'api', 'name': 'Div'},
            {'type': 'api', 'name': 'Transpose'},
            {'type': 'example', 'name': 'matmul_custom'},
            {'type': 'example', 'name': 'softmax_custom'},
        ],
    },
}

# Knowledge context - fill after running 4_joint_planning.py
KNOWLEDGE_CONTEXT = {
    "easy": "",    # TODO
    "medium": "",  # TODO
    "hard": """## API Reference

### Mmad
- **签名**: `void Mmad(const LocalTensor<DstT>& dstLocal, const LocalTensor<Src0T>& fmLocal,
    const LocalTensor<Src1T>& filterLocal, const MmadParams& mmadParams)`
- **描述**: Matrix multiplication and addition

### Muls
- **签名**: `void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)`
- **描述**: dst[i] = src[i] * scalar

### ReduceMax
- **签名**: `void ReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t mask, const int32_t repeatTimes, const int32_t srcRepStride,
    bool calIndex = 0)`
- **描述**: Index of the maximum value of all input elements

### ReduceSum
- **签名**: `void ReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t mask, const int32_t repeatTimes, const int32_t srcRepStride)`
- **描述**: sum all input elements

### Sub
- **签名**: `void Sub(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)`
- **描述**: dst = src0 - src1

### Exp
- **签名**: `void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)`
- **描述**: dst[i] = exp(src[i])

### Div
- **签名**: `void Div(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)`
- **描述**: dst = src0 / src1

### DataCopy
- **签名**: `void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const Nd2NzParams& params)`
- **描述**: Copy data between global and local memory
""",
}


def get_phase0_context(test_case: str) -> dict:
    """Get Phase 0 context for a test case."""
    ctx = PHASE0_CONTEXT.get(test_case)
    if ctx is None or ctx.get("signature") is None:
        raise ValueError(
            f"Phase 0 context not filled for '{test_case}'.\n"
            f"Please run 2_phase0.py first and fill PHASE0_CONTEXT in _config.py"
        )
    return ctx


def get_joint_plan_context(test_case: str) -> dict:
    """Get Joint Plan context for a test case."""
    plan = JOINT_PLAN_CONTEXT.get(test_case)
    if plan is None:
        raise ValueError(
            f"Joint Plan context not filled for '{test_case}'.\n"
            f"Please run 4_joint_planning.py first and fill JOINT_PLAN_CONTEXT in _config.py"
        )
    return plan


def get_knowledge_context(test_case: str) -> str:
    """Get Knowledge context for a test case."""
    return KNOWLEDGE_CONTEXT.get(test_case, "")


# =============================================================================
# Output Utilities
# =============================================================================

def ensure_output_dir(subdir: str = "") -> Path:
    """Ensure output directory exists and return path."""
    path = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path
