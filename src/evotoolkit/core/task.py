# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Task abstractions for EvoToolkit."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .solution import EvaluationResult, Solution


@dataclass
class TaskSpec:
    """Static task specification used by methods and interfaces."""

    name: str = ""
    prompt: str = ""
    modality: str = "generic"
    extras: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "TaskSpec":
        return TaskSpec(
            name=self.name,
            prompt=self.prompt,
            modality=self.modality,
            extras=dict(self.extras),
        )


class Task(ABC):
    """Abstract base class for evolutionary optimization tasks."""

    def __init__(self, data: Any):
        self.data = data
        self.spec = self.build_spec(data)

    @abstractmethod
    def build_spec(self, data: Any) -> TaskSpec:
        """Build the static task specification for this task instance."""

    @abstractmethod
    def evaluate(self, solution: Solution) -> EvaluationResult:
        """Evaluate a candidate solution."""
