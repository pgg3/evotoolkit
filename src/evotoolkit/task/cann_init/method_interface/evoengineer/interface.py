# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""EvoEngineer Interface for CANN Init Task"""

from pathlib import Path
from typing import List

from evotoolkit.core import EvaluationResult, EvoEngineerInterface, Operator, Solution

from ...cann_init_task import CANNInitTask
from .prompts import PromptMixin
from .parser import ParserMixin


class EvoEngineerCANNInterface(PromptMixin, ParserMixin, EvoEngineerInterface):
    """EvoEngineer interface for CANN kernel optimization"""

    def __init__(self, task: CANNInitTask, output_dir: str = None):
        super().__init__(task)
        self.solution_counter = 0
        # Set up projects directory
        if output_dir:
            self.projects_dir = Path(output_dir) / "projects"
            self.projects_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.projects_dir = None

    def get_init_operators(self) -> List[Operator]:
        return [Operator("init", selection_size=0)]

    def get_offspring_operators(self) -> List[Operator]:
        return [Operator("crossover", selection_size=2), Operator("mutation", selection_size=1)]

    def evaluate(self, solution: Solution) -> EvaluationResult:
        """Use evaluate_solution for CANN task"""
        return self.task.evaluate_solution(solution)

    def get_operator_prompt(
        self,
        operator_name: str,
        selected_individuals: List[Solution],
        _current_best_sol: Solution,  # Unused - CANN doesn't need current_best
        random_thoughts: List[str],
        **_kwargs,
    ) -> List[dict]:
        """Generate prompt for operators"""
        task_desc = self._get_task_description()

        if operator_name == "init":
            return self._get_init_prompt(task_desc, random_thoughts)
        elif operator_name == "crossover":
            return self._get_crossover_prompt(task_desc, selected_individuals, random_thoughts)
        elif operator_name == "mutation":
            return self._get_mutation_prompt(task_desc, selected_individuals[0], random_thoughts)
        else:
            raise ValueError(f"Unknown operator: {operator_name}")
