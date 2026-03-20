# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from dataclasses import dataclass


@dataclass
class Operator:
    """Operator definition used by EvoEngineer."""

    name: str
    selection_size: int = 0
