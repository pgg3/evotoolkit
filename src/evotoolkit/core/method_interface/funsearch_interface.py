# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from ..base_task import BaseTask
from ..solution import Solution
from .base_method_interface import BaseMethodInterface


class FunSearchInterface(BaseMethodInterface):
    """Base adapter for FunSearch algorithm"""

    def __init__(self, task: BaseTask):
        super().__init__(task)

    @abstractmethod
    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate prompt based on multiple solutions"""
        pass
