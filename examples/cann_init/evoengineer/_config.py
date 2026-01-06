# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SRC_DIR = Path(__file__).parent / "0_test_task_src"
OUTPUT_DIR = Path(__file__).parent / "output"

PYTHON_REFERENCE = (SRC_DIR / "python_reference.py").read_text()


def ensure_output_dir(subdir: str = "") -> Path:
    path = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_task_data(op_name: str = "add", npu_type: str = "Ascend910B") -> dict:
    return {
        "op_name": op_name,
        "npu_type": npu_type,
        "python_reference": PYTHON_REFERENCE,
    }
