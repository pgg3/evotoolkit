# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from ..base_task import BaseTask
from ..solution import Solution
from .base_method_interface import BaseMethodInterface


class EoHInterface(BaseMethodInterface):
    """Base interface for EoH (Evolution of Heuristics) algorithm.

    Subclasses must implement:
    - get_prompt_i1(): Generate initialization prompt
    - get_prompt_e1(solutions): Generate E1 (crossover) prompt
    - get_prompt_e2(solutions): Generate E2 (guided crossover) prompt
    - get_prompt_m1(individual): Generate M1 (mutation) prompt
    - get_prompt_m2(individual): Generate M2 (parameter mutation) prompt
    - parse_response(response): Parse LLM response into Solution
    """

    def __init__(self, task: BaseTask):
        super().__init__(task)

    @abstractmethod
    def get_prompt_i1(self) -> List[dict]:
        """Generate initialization prompt (I1 operator)."""
        pass

    @abstractmethod
    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E1 (crossover) prompt."""
        pass

    @abstractmethod
    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E2 (guided crossover) prompt."""
        pass

    @abstractmethod
    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """Generate M1 (mutation) prompt."""
        pass

    @abstractmethod
    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        """Generate M2 (parameter mutation) prompt."""
        pass
