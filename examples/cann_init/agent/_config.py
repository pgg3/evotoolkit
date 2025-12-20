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

    return HttpsApi(api_url=api_url, key=api_key, model=model, timeout=300)


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
        "signature": {'op_name': 'Relu', 'inputs': [{'name': 'x', 'dtype': 'float', 'is_tensor': True}], 'outputs': [{'name': 'output', 'dtype': 'float', 'is_tensor': True}], 'init_params': []},
        "shape_inference": {'input': '[*] (any shape)', 'output': 'same as input', 'formula': 'auto output_shape = x.sizes();'},
        "functionality": 'Applies ReLU activation max(0, x) element-wise to the input tensor, setting all negative values to zero.',
        "strategies": {'kernel': 'generate', 'tiling': 'default', 'pybind': 'default'},
    },
    "medium": {
        "op_name": "Softmax",
        "signature": None,  # TODO
        "shape_inference": None,  # TODO
        "strategies": {"tiling": "custom", "pybind": "generate"},
    },
    "hard": {
        "op_name": "SDPA",
        "signature": {'op_name': 'SDPA', 'inputs': [{'name': 'q', 'dtype': 'float', 'is_tensor': True}, {'name': 'k', 'dtype': 'float', 'is_tensor': True}, {'name': 'v', 'dtype': 'float', 'is_tensor': True}], 'outputs': [{'name': 'output', 'dtype': 'float', 'is_tensor': True}], 'init_params': []},
        "shape_inference": {'input': 'Q=[B, S, D], K=[B, S, D], V=[B, S, D] where B=batch, S=seq_len, D=d_model', 'output': '[B, S, D] (same as Q, K, V)', 'formula': 'auto output_shape = q.sizes();'},
        "functionality": 'Implements scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V, computing attention-weighted values where queries attend to keys and retrieve values.',
        "strategies": {'kernel': 'generate', 'tiling': 'generate', 'pybind': 'generate'},
    },
}

# Joint Plan output - fill after running 4_joint_planning.py
JOINT_PLAN_CONTEXT = {
    "easy": {
        'tiling_proposal': '## Reasoning\nReLU applies element-wise activation (max(0, x)) to a single tensor. Single input, output shape equals input shape.\n\nStrategy: default',
        'kernel_design': '## Reasoning\nReLU is element-wise activation (max(0, x)) on single tensor. Default template handles this correctly.\n\naccepted: true\nstrategy: default\n\n## Kernel Design\n- Paradigm: vector\n- Pipeline: double_buffer\n- Operations: [Relu]\n\n## Kernel Pseudocode\n```cpp\n// Default template: totalLength, tileNum from TilingData\n// Multi-core: GetBlockIdx() computes core offset\nfor (int i = 0; i < tileNum; i++) {\n    uint32_t coreIdx = GetBlockIdx();\n    uint32_t globalTileIdx = coreIdx * tileNum + i;\n    uint32_t offset = globalTileIdx * tileLength;\n    \n    CopyIn: GM -> UB, x[offset : offset + tileLength]\n    Compute: y = Relu(x)  // element-wise max(0, x)\n    CopyOut: UB -> GM, output[offset : offset + tileLength]\n}\n```\n\n## Useful References\n- APIs: [Relu, DataCopy, GetBlockIdx]\n- Examples: [relu_custom, abs_custom]',
        'tiling_strategy': 'default',
        'tiling_fields': [],
        'kernel_pseudocode': '''// Default template: totalLength, tileNum from TilingData
// Multi-core: GetBlockIdx() computes core offset
for (int i = 0; i < tileNum; i++) {
    uint32_t coreIdx = GetBlockIdx();
    uint32_t globalTileIdx = coreIdx * tileNum + i;
    uint32_t offset = globalTileIdx * tileLength;

    CopyIn: GM -> UB, x[offset : offset + tileLength]
    Compute: y = Relu(x)  // element-wise max(0, x)
    CopyOut: UB -> GM, output[offset : offset + tileLength]
}''',
        'tiling_execution': None,
        'retrieval_requests': [
            {'type': 'api', 'name': 'Relu'},
            {'type': 'api', 'name': 'DataCopy'},
            {'type': 'api', 'name': 'GetBlockIdx'},
            {'type': 'example', 'name': 'relu_custom'},
            {'type': 'example', 'name': 'abs_custom'},
        ],
    },
    "medium": None,  # TODO
    "hard": {
        'tiling_strategy': 'custom',
        'tiling_fields': [
            {'name': 'batchSize', 'type': 'uint32_t', 'purpose': 'Total batch size (32)'},
            {'name': 'seqLen', 'type': 'uint32_t', 'purpose': 'Sequence length (128)'},
            {'name': 'dModel', 'type': 'uint32_t', 'purpose': 'Model dimension (768)'},
            {'name': 'seqTileSize', 'type': 'uint32_t', 'purpose': 'Tile size for sequence dimension (32)'},
            {'name': 'dTileSize', 'type': 'uint32_t', 'purpose': 'Tile size for model dimension (256)'},
            {'name': 'batchesPerCore', 'type': 'uint32_t', 'purpose': 'Batches assigned per core: ceil(batchSize/24)'},
            {'name': 'seqTiles', 'type': 'uint32_t', 'purpose': 'Number of sequence tiles: seqLen/seqTileSize (4)'},
            {'name': 'dTiles', 'type': 'uint32_t', 'purpose': 'Number of d_model tiles: ceil(dModel/dTileSize) (3)'},
        ],
        'kernel_pseudocode': '''// Using tiling fields from proposal
// Per-core variables
uint32_t batchStart = GetBlockIdx() * batchesPerCore;
uint32_t batchEnd = min(batchStart + batchesPerCore, batchSize);

// Allocate UB buffers
LocalTensor<float> Q_tile[2]; // [seqTileSize, dTileSize] - double buffer
LocalTensor<float> K_tile[2]; // [seqLen, dTileSize] - double buffer
LocalTensor<float> V_tile[2]; // [seqTileSize, dTileSize] - double buffer
LocalTensor<float> Scores_row; // [seqTileSize, seqLen] - full row
LocalTensor<float> Output_tile; // [seqTileSize, dModel]

for (uint32_t b = batchStart; b < batchEnd; b++) {
    for (uint32_t i_tile = 0; i_tile < seqTiles; i_tile++) {
        uint32_t i_start = i_tile * seqTileSize;

        // Stage 1: Compute Scores[i_tile, :] via reduction over D
        InitBuffer(Scores_row, 0.0f);
        for (uint32_t d_tile = 0; d_tile < dTiles; d_tile++) {
            // Load Q tile and K tile, accumulate Scores_row += Q_tile @ K_tile^T
            MatMul(Scores_row, Q_tile, K_tile, accumulate=true, transposeB=true);
        }

        // Stage 2: Scale and Softmax on full row
        VecMuls(Scores_row, Scores_row, 1.0f / sqrt(dModel));
        Softmax(Scores_row);

        // Stage 3: Compute Output[i_tile, :] via tiled matmul
        InitBuffer(Output_tile, 0.0f);
        for (uint32_t d_tile = 0; d_tile < dTiles; d_tile++) {
            for (uint32_t s_tile = 0; s_tile < seqTiles; s_tile++) {
                // Load V tile, accumulate output
                MatMul(Output_tile, Scores_slice, V_tile, accumulate=true);
            }
        }

        CopyUB2GM(Output[b, i_start:i_start+seqTileSize, :], Output_tile);
    }
}''',
        'tiling_execution': '''for batch in myBatches:  // ~1-2 batches per core
    for i_tile in range(S/32):  // 4 tiles (output rows)
        // Stage 1: Compute full Scores row via tiled matmul over D
        Scores_row[32, 128] = zeros
        for d_tile in range(D/256):  // 3 tiles
            CopyIn: Q[batch, i_tile*32:(i_tile+1)*32, d_tile*256:(d_tile+1)*256]
            CopyIn: K[batch, :, d_tile*256:(d_tile+1)*256]  // full S=128
            Compute: Scores_row += Cube(Q_tile, K_tile^T)

        // Stage 2: Scale and Softmax on full row
        Compute: Scores_row = Scores_row / sqrt(768)
        Compute: Scores_row = Softmax(Scores_row, dim=-1)

        // Stage 3: Compute output via tiled matmul over D and S
        Output_tile[32, D] = zeros
        for d_tile in range(D/256):  // 3 tiles
            for s_tile in range(S/32):  // 4 tiles
                CopyIn: V[batch, s_tile*32:(s_tile+1)*32, d_tile*256:(d_tile+1)*256]
                Compute: Output_tile[:, d_tile*256:(d_tile+1)*256] += Cube(Scores_row[:, s_tile*32:(s_tile+1)*32], V_tile)

        CopyOut: Output[batch, i_tile*32:(i_tile+1)*32, :]''',
        'retrieval_requests': [
            {'type': 'api', 'name': 'DataCopy'},
            {'type': 'api', 'name': 'MatMul'},
            {'type': 'api', 'name': 'Muls'},
            {'type': 'api', 'name': 'Softmax'},
            {'type': 'api', 'name': 'Add'},
            {'type': 'example', 'name': 'Attention'},
            {'type': 'example', 'name': 'MatMul'},
            {'type': 'example', 'name': 'Softmax'},
            {'type': 'example', 'name': 'LayerNorm'},
        ],
    },
}

# Knowledge context - fill after running 4_joint_planning.py
KNOWLEDGE_CONTEXT = {
    "easy": """## API Reference

### Relu
- **Signature**: `void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)`
- **Description**: dst[i] = (src[i] < 0) ? 0 : src[i]

### DataCopy
- **Signature**: `void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                     const Nd2NzParams& intriParams)`
- **Description**: format transform(such as nd2nz) during data load from OUT to L1

### SetFixPipeConfig
- **Signature**: `void SetFixPipeConfig(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
    bool isUnitFlag = false)`
- **Description**: After calculation, process the results

## Example Reference

### relu
**Purpose**: This is the **exact same operator** as the current task. ReLU implements the element-wise max(0, x) operation, which is precisely what needs to be implemented. This example provides the direct reference implementation.

**Mapping to Current Task**:
- Example's `x` input tensor → Current task's `x` input tensor (dtype: float)
- Example's `output` tensor → Current task's `output` tensor (dtype: float)
- Example's ReLU computation (max(0, x)) → Current task's ReLU computation (max(0, x))
- Example's tiling strategy → Current task's tiling strategy for dividing workload across cores
- Example's multi-core execution pattern → Current task's multi-core execution with GetBlockIdx()

**Implementation Patterns**:
- Data flow: GM → UB (CopyIn via DataCopy) → Compute (Relu intrinsic) → UB → GM (CopyOut via DataCopy)
- Pipeline: Likely uses double buffer for overlapping data transfer and computation
- API sequence:

**Key Techniques**:
- Direct use of Ascend C Relu intrinsic/API for element-wise max(0, x) operation
- Tile-based processing to handle data larger than UB (Unified Buffer) capacity
- Block-level parallelism with GetBlockIdx() for multi-core execution
- Sequential tile processing w
""",
    "medium": "",  # TODO
    "hard": """## API Reference

### Gemm
- **Signature**: `void Gemm(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, const uint32_t m, const uint32_t k, const uint32_t n, GemmTiling tilling,
    bool partialsum = true, int32_t initValue = 0)`
- **Description**: Multiply two matrices

### DataCopy
- **Signature**: `void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                     const Nd2NzParams& intriParams)`
- **Description**: format transform(such as nd2nz) during data load from OUT to L1

### Muls
- **Signature**: `void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)`
- **Description**: dst[i] = src[i] * scalar

### Exp
- **Signature**: `void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)`
- **Description**: dst[i] = exp(src[i])

### ReduceSum
- **Signature**: `void ReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t mask, const int32_t repeatTimes, const int32_t srcRepStride)`
- **Description**: sum all input elements

### Div
- **Signature**: `void Div(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams)`
- **Description**: dst = src0 / src1

### ReduceMax
- **Signature**: `void ReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t mask, const int32_t repeatTimes, const int32_t srcRepStride,
    bool calIndex = 0)`
- **Description**: Index of the maximum value of all input elements

## Example Reference

### softmax_custom
**Purpose**: Shows implementation of row-wise softmax operation

**Key Techniques**:
- ReduceMax for numerical stability
- Exp and ReduceSum for softmax computation
- Row-wise processing pattern

### matmul_leakyrelu_custom
**Purpose**: Shows tiled matrix multiplication with fused activation

**Key Techniques**:
- Cube unit for matrix multiplication
- Double buffering for data movement
- Tiled computation pattern
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
