# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Utilities for CANN Init: templates, backend, pybind."""

from .templates import AscendCTemplateGenerator
from .backend import (
    ascend_compile,
    ascend_setup,
    ascend_build,
    write_project_files,
    execute_correctness_check,
    set_seed,
    measure_performance,
    CANNSandboxExecutor,
)

__all__ = [
    "AscendCTemplateGenerator",
    "ascend_compile",
    "ascend_setup",
    "ascend_build",
    "write_project_files",
    "execute_correctness_check",
    "set_seed",
    "measure_performance",
    "CANNSandboxExecutor",
]
