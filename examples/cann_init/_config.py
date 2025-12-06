# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Shared configuration for CANN Init examples.

All example scripts import test data and utilities from here.
"""

from pathlib import Path

# Directories
SRC_DIR = Path(__file__).parent / "0_test_task_src"
OUTPUT_DIR = Path(__file__).parent / "output"

# Load test data
PYTHON_REFERENCE = (SRC_DIR / "python_reference.py").read_text()
KERNEL_SRC = (SRC_DIR / "kernel_src.cpp").read_text()

# Load tiling configuration
_tiling_config = {}
exec((SRC_DIR / "tiling_config.py").read_text(), _tiling_config)
TILING_FIELDS = _tiling_config["TILING_FIELDS"]
TILING_FUNC_BODY = _tiling_config["TILING_FUNC_BODY"]
BLOCK_DIM = _tiling_config["BLOCK_DIM"]


def ensure_output_dir(subdir: str = "") -> Path:
    """Ensure output directory exists and return path."""
    path = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_task_data(op_name: str = "add", npu_type: str = "Ascend910B") -> dict:
    """Get standard task data dict."""
    return {
        "op_name": op_name,
        "python_reference": PYTHON_REFERENCE,
        "npu_type": npu_type,
    }
