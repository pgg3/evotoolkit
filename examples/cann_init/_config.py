# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Shared configuration for CANN Init examples.

All example scripts import test data and utilities from here.
Loads environment variables from .env file automatically.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv(Path(__file__).parent / ".env")

# Directories
SRC_DIR = Path(__file__).parent / "0_test_task_src"
OUTPUT_DIR = Path(__file__).parent / "output"

# ============================================================================
# Test Data Constants - for scripts 1-4 (Add operator example)
# ============================================================================

# Load from source files
PYTHON_REFERENCE = (SRC_DIR / "python_reference.py").read_text()
KERNEL_SRC = (SRC_DIR / "kernel_src.cpp").read_text()

# Default mode config
BLOCK_DIM = 8

# Full LLM mode config - load from tiling_config.py
_tiling_config_path = SRC_DIR / "tiling_config.py"
_tiling_vars = {}
exec(_tiling_config_path.read_text(), _tiling_vars)

HOST_TILING_SRC = _tiling_vars["HOST_TILING_SRC"]
HOST_OPERATOR_SRC = _tiling_vars["HOST_OPERATOR_SRC"]
PYTHON_BIND_SRC = _tiling_vars["PYTHON_BIND_SRC"]

# ============================================================================
# Utility Functions
# ============================================================================


def ensure_output_dir(subdir: str = "") -> Path:
    """Ensure output directory exists and return path."""
    path = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_task_data(op_name: str = "add", npu_type: str = "Ascend910B") -> dict:
    """
    Get task data dict for CANNInitTask (scripts 3-4).

    Returns:
        dict with op_name, npu_type, python_reference
    """
    return {
        "op_name": op_name,
        "npu_type": npu_type,
        "python_reference": PYTHON_REFERENCE,
    }


def get_llm():
    """Get LLM instance from environment variables (script 5)."""
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


def get_task(op_name: str = "Add", npu_type: str = "Ascend910B"):
    """Get CANNInitTask instance (script 5)."""
    from evotoolkit.task.cann_init import CANNInitTask

    return CANNInitTask(op_name=op_name, npu_type=npu_type)
