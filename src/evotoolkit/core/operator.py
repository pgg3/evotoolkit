# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Operator definitions for evolutionary algorithms.

This module contains operator classes used by evolutionary algorithms
to define genetic operations and their parameters.
"""


class Operator:
    """Simple operator class with name and selection size."""

    def __init__(self, name: str, selection_size: int = 0):
        self.name = name
        self.selection_size = selection_size
