# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from dataclasses import dataclass, field
from typing import Any

from evotoolkit.core import MethodState, Solution


@dataclass
class FunSearchState(MethodState):
    """Runtime state for FunSearch."""

    tot_sample_nums: int = 0
    batch_size: int = 1
    current_batch_solutions: list[Solution] = field(default_factory=list)
    current_batch_usage: list[dict] = field(default_factory=list)
    current_batch_start: int = 0
    programs_database: Any = None
    best_per_batch: list[dict] = field(default_factory=list)

    @property
    def current_batch_id(self) -> int:
        if self.tot_sample_nums <= 0:
            return 0
        return (self.tot_sample_nums - 1) // self.batch_size
