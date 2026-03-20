# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from abc import abstractmethod
from typing import List

from evotoolkit.core import MethodInterface, Solution, Task


class EoHInterface(MethodInterface):
    """Base adapter for EoH (Evolution of Heuristics) algorithm."""

    def __init__(self, task: Task):
        super().__init__(task)

    @abstractmethod
    def get_prompt_i1(self) -> List[dict]:
        """Generate initialization prompt (I1 operator)."""

    @abstractmethod
    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E1 (crossover) prompt."""

    @abstractmethod
    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E2 (guided crossover) prompt."""

    @abstractmethod
    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """Generate M1 (mutation) prompt."""

    @abstractmethod
    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        """Generate M2 (parameter mutation) prompt."""
