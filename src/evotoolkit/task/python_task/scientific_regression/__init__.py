# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Scientific Symbolic Regression Task for EvoToolkit.

This module provides tasks for discovering mathematical functions from
real scientific datasets including:
- Bacterial growth modeling (bactgrow)
- Physical oscillation systems (oscillator1, oscillator2)
- Material stress-strain relationships (stressstrain)
"""

from .scientific_regression_task import ScientificRegressionTask

__all__ = ['ScientificRegressionTask']
