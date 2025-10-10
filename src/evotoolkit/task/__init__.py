# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Task module for evolutionary optimization.
"""

from .python_task import (
    EoHPythonInterface,
    EvoEngineerPythonInterface,
    FunSearchPythonInterface,
    PythonTask,
)
from .string_optimization import (
    EoHStringInterface,
    EvoEngineerStringInterface,
    FunSearchStringInterface,
    PromptOptimizationTask,
    StringTask,
)

__all__ = [
    "PythonTask",
    "EoHPythonInterface",
    "FunSearchPythonInterface",
    "EvoEngineerPythonInterface",
    "StringTask",
    "PromptOptimizationTask",
    "EvoEngineerStringInterface",
    "EoHStringInterface",
    "FunSearchStringInterface",
]
