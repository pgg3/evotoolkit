# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""EvoEngineer Interface for CANN Init Task"""

from typing import List

from evotoolkit.core import EvaluationResult, EvoEngineerInterface, Operator, Solution

from ...cann_init_task import CANNInitTask
from .prompts import PromptMixin
from .parser import ParserMixin


class EvoEngineerCANNInterface(PromptMixin, ParserMixin, EvoEngineerInterface):
    """EvoEngineer interface for CANN kernel optimization"""

    def __init__(self, task: CANNInitTask):
        super().__init__(task)
        self.valid_require = 2

    def get_init_operators(self) -> List[Operator]:
        return [Operator("init", selection_size=0)]

    def get_offspring_operators(self) -> List[Operator]:
        return [Operator("crossover", selection_size=2), Operator("mutation", selection_size=1)]

    def make_init_sol(self) -> Solution:
        """Empty solution - let LLM generate from scratch"""
        return Solution("", other_info={"name": "empty", "thought": ""})

    def evaluate(self, solution: Solution) -> EvaluationResult:
        """Use evaluate_solution for CANN task"""
        return self.task.evaluate_solution(solution)

    def get_operator_prompt(
        self,
        operator_name: str,
        selected_individuals: List[Solution],
        current_best_sol: Solution,
        random_thoughts: List[str],
        **_kwargs,
    ) -> List[dict]:
        """Generate prompt for operators"""
        task_desc = self._get_task_description()

        if operator_name == "init":
            return self._get_init_prompt(task_desc, current_best_sol, random_thoughts)
        elif operator_name == "crossover":
            return self._get_crossover_prompt(task_desc, selected_individuals, current_best_sol, random_thoughts)
        elif operator_name == "mutation":
            return self._get_mutation_prompt(task_desc, selected_individuals[0], current_best_sol, random_thoughts)
        else:
            raise ValueError(f"Unknown operator: {operator_name}")
