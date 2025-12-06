# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANN Init task module for Ascend C operator generation.

This module provides tools for generating and evaluating Ascend C operators
from Python reference implementations.

Components:
- CANNInitTask: Main task class for operator evaluation
- AscendCEvaluator: Handles compilation, deployment, and verification
- AscendCTemplateGenerator: Generates boilerplate code from templates
- OperatorSignatureParser: Parses Python code to extract operator signature

Data Structures:
- CompileResult: Compilation output that can be saved/loaded
- CANNSolutionConfig: Typed wrapper for Solution.other_info

Backend utilities (adapted from MultiKernelBench):
- ascend_compile: Compilation pipeline
- execute_correctness_check: Correctness verification
- measure_performance: Performance measurement
"""

from .cann_init_task import CANNInitTask
from .evaluator import AscendCEvaluator
from .templates import AscendCTemplateGenerator
from .signature_parser import OperatorSignatureParser
from .data_structures import CompileResult, CANNSolutionConfig

# Backend utilities
from .backend import (
    ascend_compile,
    execute_correctness_check,
    measure_performance,
)

__all__ = [
    # Main classes
    "CANNInitTask",
    "AscendCEvaluator",
    "AscendCTemplateGenerator",
    "OperatorSignatureParser",
    # Data structures
    "CompileResult",
    "CANNSolutionConfig",
    # Backend utilities
    "ascend_compile",
    "execute_correctness_check",
    "measure_performance",
]
