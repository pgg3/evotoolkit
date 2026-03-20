# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Core runtime types for EvoToolkit."""

from .interface import MethodInterface
from .method import Method
from .solution import EvaluationResult, Solution, SolutionMetadata
from .state import MethodState, PopulationState
from .store import RunStore
from .task import Task, TaskSpec

__all__ = [
    "Solution",
    "SolutionMetadata",
    "EvaluationResult",
    "Task",
    "TaskSpec",
    "Method",
    "MethodState",
    "PopulationState",
    "RunStore",
    "MethodInterface",
]
