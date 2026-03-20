# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from evotoolkit.core import MethodInterface, Solution, Task


class FunSearchInterface(MethodInterface):
    """Base adapter for FunSearch."""

    def __init__(self, task: Task):
        super().__init__(task)

    @abstractmethod
    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate prompt based on multiple solutions."""
