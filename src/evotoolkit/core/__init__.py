# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Core runtime types for EvoToolkit."""

from .base_method import Method
from .base_task import BaseTask
from .method_interface import (
    BaseMethodInterface,
    EoHInterface,
    EvoEngineerInterface,
    FunSearchInterface,
)
from .method_state import MethodState, PopulationMethodState
from .operator import Operator
from .run_store import RunStore
from .solution import EvaluationResult, Solution

__all__ = [
    "Solution",
    "EvaluationResult",
    "Operator",
    "BaseTask",
    "Method",
    "MethodState",
    "PopulationMethodState",
    "RunStore",
    "BaseMethodInterface",
    "EoHInterface",
    "FunSearchInterface",
    "EvoEngineerInterface",
]
