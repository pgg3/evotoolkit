# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Core runtime types for EvoToolkit."""

from .interface import MethodInterface
from .method import IterativeMethod, Method, PopulationMethod
from .solution import EvaluationResult, Solution, SolutionMetadata
from .state import IterationState, MethodState, PopulationState
from .store import RunStore
from .task import Task, TaskSpec

__all__ = [
    "Solution",
    "SolutionMetadata",
    "EvaluationResult",
    "Task",
    "TaskSpec",
    "Method",
    "IterativeMethod",
    "PopulationMethod",
    "MethodState",
    "IterationState",
    "PopulationState",
    "RunStore",
    "MethodInterface",
]
