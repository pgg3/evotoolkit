# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
EvoTool Core Module

This module contains the fundamental data structures and base classes
used throughout the EvoTool framework. It provides the foundation
for all evolutionary algorithms and task implementations.

Core Components:
- Solution, EvaluationResult: Basic data structures
- Operator: Evolutionary operator definitions
- BaseTask: Abstract task interface (unified evaluator and config)
- BaseConfig: Base configuration class
- Method: Base evolutionary method class
- BaseRunStateDict: Base run state dictionary for saving/loading state
- HistoryManager: Manager for evolution history
- method_interface: Interfaces for different evolutionary algorithms
"""

from .solution import Solution, EvaluationResult
from .operator import Operator
from .base_task import BaseTask
from .base_config import BaseConfig
from .base_method import Method
from .base_run_state_dict import BaseRunStateDict
from .history_manager import HistoryManager

# Import method interfaces
from .method_interface import (
    BaseMethodInterface,
    EoHInterface,
    FunSearchInterface,
    EvoEngineerInterface
)

__all__ = [
    'Solution',
    'EvaluationResult',
    'Operator',
    'BaseTask',
    'BaseConfig',
    'Method',
    'BaseRunStateDict',
    'HistoryManager',
    'BaseMethodInterface',
    'EoHInterface',
    'FunSearchInterface',
    'EvoEngineerInterface',
]