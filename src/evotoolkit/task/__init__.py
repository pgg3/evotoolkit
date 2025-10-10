# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Task module for evolutionary optimization.
"""

from .python_task import (
    PythonTask,
    EoHPythonInterface,
    FunSearchPythonInterface,
    EvoEngineerPythonInterface
)

from .string_optimization import (
    StringTask,
    PromptOptimizationTask,
    EvoEngineerStringInterface,
    EoHStringInterface,
    FunSearchStringInterface,
)

__all__ = [
    'PythonTask',
    'EoHPythonInterface',
    'FunSearchPythonInterface',
    'EvoEngineerPythonInterface',
    'StringTask',
    'PromptOptimizationTask',
    'EvoEngineerStringInterface',
    'EoHStringInterface',
    'FunSearchStringInterface',
]
