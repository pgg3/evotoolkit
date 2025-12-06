# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Backend utilities for Ascend C operator evaluation.

Adapted from MultiKernelBench with modifications for evotoolkit integration.
"""

from .ascend_compile import ascend_compile
from .correctness import execute_correctness_check, set_seed
from .performance import measure_performance
from .sandbox import CANNSandboxExecutor

__all__ = [
    "ascend_compile",
    "execute_correctness_check",
    "set_seed",
    "measure_performance",
    "CANNSandboxExecutor",
]
