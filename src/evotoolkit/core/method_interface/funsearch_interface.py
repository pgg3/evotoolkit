# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from ..base_task import BaseTask
from ..solution import Solution
from .base_method_interface import BaseMethodInterface


class FunSearchInterface(BaseMethodInterface):
    """Base interface for FunSearch algorithm.

    Subclasses must implement:
    - get_prompt(solutions): Generate prompt based on multiple solutions
    - parse_response(response): Parse LLM response into Solution
    """

    def __init__(self, task: BaseTask):
        super().__init__(task)

    @abstractmethod
    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate prompt based on multiple solutions."""
        pass
