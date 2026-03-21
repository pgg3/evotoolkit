# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from evotoolkit.core import MethodInterface, Solution, Task

from .operator import Operator


class EvoEngineerInterface(MethodInterface):
    """Base adapter for EvoEngineer."""

    def __init__(self, task: Task):
        super().__init__(task)
        self.valid_require = 2

    @staticmethod
    def _format_solution_score(solution: Solution) -> str:
        if solution.evaluation_res is None or solution.evaluation_res.score is None:
            return "None"
        return f"{solution.evaluation_res.score:.5f}"

    @abstractmethod
    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for this task."""

    @abstractmethod
    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for this task."""

    @abstractmethod
    def get_operator_prompt(
        self,
        operator_name: str,
        selected_individuals: List[Solution],
        current_best_sol: Solution | None,
        random_descriptions: List[str],
        **kwargs,
    ) -> List[dict]:
        """Generate prompt for a named operator."""
