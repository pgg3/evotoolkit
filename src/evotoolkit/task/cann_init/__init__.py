# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANN Init: Ascend C operator generation and evaluation."""

from .cann_init_task import CANNInitTask
from .evaluator import AscendCEvaluator
from .utils.templates import AscendCTemplateGenerator
from .signature_parser import OperatorSignatureParser
from .data_structures import CompileResult, CANNSolutionConfig
from .method_interface import CANNIniterInterface, EvoEngineerCANNInterface
from .utils.backend import ascend_compile, execute_correctness_check, measure_performance

__all__ = [
    "CANNInitTask",
    "CANNIniterInterface",
    "EvoEngineerCANNInterface",
    "AscendCEvaluator",
    "AscendCTemplateGenerator",
    "OperatorSignatureParser",
    "CompileResult",
    "CANNSolutionConfig",
    "ascend_compile",
    "execute_correctness_check",
    "measure_performance",
]
