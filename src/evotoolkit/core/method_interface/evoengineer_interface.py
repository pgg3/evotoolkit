# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from evotoolkit.core import BaseTask, Operator, Solution

from .base_method_interface import BaseMethodInterface


class EvoEngineerInterface(BaseMethodInterface):
    """Base adapter for EvoEngineer algorithm"""

    def __init__(self, task: BaseTask):
        super().__init__(task)
        self.valid_require = 2

    def make_init_sol(self) -> Solution:
        init_sol = self.task.make_init_sol_wo_other_info()
        other_info = {"name": "Baseline", "thought": "Baseline"}
        init_sol.other_info = other_info
        return init_sol

    @abstractmethod
    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for this task (should have selection_size=0)"""
        pass

    @abstractmethod
    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for this task"""
        pass

    @abstractmethod
    def get_operator_prompt(
        self,
        operator_name: str,
        selected_individuals: List[Solution],
        current_best_sol: Solution,
        random_thoughts: List[str],
        **kwargs,
    ) -> List[dict]:
        """Generate prompt for any operator

        Args:
            operator_name: Name of the operator
            selected_individuals: Selected individuals for the operator
            **kwargs: Additional operator-specific parameters
            current_best_sol:
            random_thoughts:
        """
        pass
