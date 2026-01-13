# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from ..base_task import BaseTask
from ..solution import EvaluationResult, Solution
from ..operator import Operator
from .base_method_interface import BaseMethodInterface


class EvoEngineerInterface(BaseMethodInterface):
    """Base interface for EvoEngineer algorithm.

    Subclasses must implement:
    - get_init_operators(): Return initialization operators
    - get_offspring_operators(): Return offspring operators
    - get_operator_prompt(): Generate prompt for any operator
    - parse_response(): Parse LLM response into Solution
    """

    def __init__(self, task: BaseTask):
        super().__init__(task)

    def evaluate(self, solution: Solution) -> EvaluationResult:
        """Evaluate a solution. Override for tasks requiring evaluate_solution."""
        return self.task.evaluate_code(solution.sol_string)

    @abstractmethod
    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for this task (should have selection_size=0)."""
        pass

    @abstractmethod
    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for this task."""
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
        """Generate prompt for any operator.

        Args:
            operator_name: Name of the operator
            selected_individuals: Selected individuals for the operator
            current_best_sol: Current best solution
            random_thoughts: Random thoughts from population
            **kwargs: Additional operator-specific parameters
        """
        pass
