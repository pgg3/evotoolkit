# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from dataclasses import dataclass, field
from typing import Any

from evotoolkit.core import MethodState, Solution


@dataclass
class FunSearchState(MethodState):
    """Runtime state for FunSearch."""

    sample_count: int = 0
    batch_size: int = 1
    current_batch_solutions: list[Solution] = field(default_factory=list)
    current_batch_usage: list[dict] = field(default_factory=list)
    current_batch_start: int = 0
    programs_database: Any = None
    best_per_batch: list[dict] = field(default_factory=list)

    @property
    def current_batch_id(self) -> int:
        if self.sample_count <= 0:
            return 0
        return (self.sample_count - 1) // self.batch_size

    def get_progress_index(self) -> int:
        return self.current_batch_id

    def get_sample_count(self) -> int:
        return self.sample_count
