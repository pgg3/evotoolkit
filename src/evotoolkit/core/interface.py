# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Method/task adapter abstractions."""

import abc
from abc import abstractmethod
from typing import Any

from .solution import Solution, SolutionMetadata
from .task import Task


class MethodInterface(abc.ABC):
    """Base adapter between a method runtime and a task."""

    def __init__(self, task: Task):
        self.task = task

    @staticmethod
    def make_solution(
        sol_string: str,
        *,
        name: str = "",
        description: str = "",
        extras: dict[str, Any] | None = None,
    ) -> Solution:
        metadata = SolutionMetadata(
            name=name,
            description=description,
            extras=dict(extras or {}),
        )
        return Solution(sol_string, metadata=metadata)

    @abstractmethod
    def parse_response(self, response_str: str) -> Solution:
        """Convert raw LLM output into a candidate Solution."""
