# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from evotoolkit.core import MethodInterface, Solution, SolutionMetadata, Task

from .operator import Operator


class EvoEngineerInterface(MethodInterface):
    """Base adapter for EvoEngineer."""

    def __init__(self, task: Task):
        super().__init__(task)
        self.valid_require = 2

    def _make_initial_solution(self) -> Solution:
        if not self.task.spec.initial_solution.strip():
            raise ValueError("Task spec must define initial_solution")
        init_sol = Solution(
            self.task.spec.initial_solution,
            metadata=SolutionMetadata(
                name=self.task.spec.initial_name,
                description=self.task.spec.initial_description,
                extras=dict(self.task.spec.initial_extras),
            ),
        )
        init_sol.metadata = init_sol.metadata.with_defaults(name="Initial", description="Initial")
        return init_sol

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
        current_best_sol: Solution,
        random_descriptions: List[str],
        **kwargs,
    ) -> List[dict]:
        """Generate prompt for a named operator."""
