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
CONTEXTS_DIR = Path(__file__).parent / "contexts"  # Saved contexts from pipeline runs


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
# Shared Context
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


# =============================================================================
# Context Loading from Files (saved by 4_joint_planning.py)
# =============================================================================

import json

def _load_json_context(test_case: str, filename: str) -> dict:
    """Load JSON context from contexts/{test_case}/{filename}."""
    file_path = CONTEXTS_DIR / test_case / filename
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_text_context(test_case: str, filename: str) -> str:
    """Load text context from contexts/{test_case}/{filename}."""
    file_path = CONTEXTS_DIR / test_case / filename
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_phase0_context(test_case: str) -> dict:
    """Get Phase 0 context for a test case.

    Priority:
    1. Load from contexts/{test_case}/phase0_context.json (if exists)
    2. Fallback to built-in PHASE0_CONTEXT
    """
    # Try loading from file first
    file_ctx = _load_json_context(test_case, "phase0_context.json")
    if file_ctx and file_ctx.get("signature"):
        return file_ctx

    # Fallback to built-in context
    ctx = PHASE0_CONTEXT.get(test_case)
    if ctx is None or ctx.get("signature") is None:
        raise ValueError(
            f"Phase 0 context not found for '{test_case}'.\n"
            f"Please run 2_phase0.py first to generate contexts/{test_case}/phase0_context.json"
        )
    return ctx


def get_joint_plan_context(test_case: str) -> dict:
    """Get Joint Plan context for a test case (loaded from contexts/{test_case}/joint_plan.json)."""
    plan = _load_json_context(test_case, "joint_plan.json")
    if not plan:
        raise ValueError(
            f"Joint Plan context not found for '{test_case}'.\n"
            f"Please run 4_joint_planning.py first to generate contexts/{test_case}/joint_plan.json"
        )
    return plan


def get_knowledge_context(test_case: str) -> str:
    """Get Knowledge context for a test case (loaded from contexts/{test_case}/knowledge_context.md)."""
    return _load_text_context(test_case, "knowledge_context.md")


def get_knowledge_summary(test_case: str) -> dict:
    """Get Knowledge summary for a test case (loaded from contexts/{test_case}/knowledge_summary.json).

    Returns:
        {
            "api_summaries": [...],
            "example_summaries": [...],
        }
    """
    return _load_json_context(test_case, "knowledge_summary.json")


def get_pybind_context(test_case: str) -> dict:
    """Get Pybind context for a test case (loaded from contexts/{test_case}/pybind_context.json).

    Returns:
        {
            "has_pybind_src": bool,
            "shape_inference_code": str or None,
        }
    """
    return _load_json_context(test_case, "pybind_context.json")


# =============================================================================
# Output Utilities
# =============================================================================

def ensure_output_dir(subdir: str = "") -> Path:
    """Ensure output directory exists and return path."""
    path = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path
