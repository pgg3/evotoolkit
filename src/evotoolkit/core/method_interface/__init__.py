# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Method interfaces for different evolutionary algorithms.

This module contains interface classes that define how different evolutionary
algorithms interact with tasks.
"""

from .base_method_interface import BaseMethodInterface
from .eoh_interface import EoHInterface
from .funsearch_interface import FunSearchInterface
from .evoengineer_interface import EvoEngineerInterface

__all__ = [
    'BaseMethodInterface',
    'EoHInterface',
    'FunSearchInterface',
    'EvoEngineerInterface',
]
