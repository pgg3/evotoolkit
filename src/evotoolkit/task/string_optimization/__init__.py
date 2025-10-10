# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
String optimization tasks for evolutionary optimization.

This module provides tasks for optimizing strings, such as prompts,
templates, and configurations, using LLM-driven evolution.
"""

from .method_interface import (
    EoHStringInterface,
    EvoEngineerStringInterface,
    FunSearchStringInterface,
)
from .prompt_optimization import PromptOptimizationTask
from .string_task import StringTask

__all__ = [
    "StringTask",
    "PromptOptimizationTask",
    "EvoEngineerStringInterface",
    "EoHStringInterface",
    "FunSearchStringInterface",
]
