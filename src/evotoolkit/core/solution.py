# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Core data structures for EvoToolkit.

This module contains the fundamental data structures used throughout the framework:
- Solution: Represents a candidate solution with evaluation results
- SolutionMetadata: Typed metadata shared across methods and tasks
- EvaluationResult: Stores evaluation outcome and metrics
"""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class SolutionMetadata:
    """Typed metadata carried alongside a candidate solution."""

    name: str = ""
    description: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, metadata: "SolutionMetadata | Mapping[str, Any] | None") -> "SolutionMetadata":
        if metadata is None:
            return cls()
        if isinstance(metadata, cls):
            return cls(
                name=metadata.name,
                description=metadata.description,
                extras=dict(metadata.extras),
            )
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Unsupported solution metadata type: {type(metadata)!r}")

        payload = dict(metadata)
        name = payload.pop("name", "")
        description = payload.pop("description", payload.pop("thought", payload.pop("algorithm", "")))
        return cls(
            name="" if name is None else str(name),
            description="" if description is None else str(description),
            extras=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extras)
        if self.name:
            payload["name"] = self.name
        if self.description:
            payload["description"] = self.description
        return payload

    def with_defaults(self, *, name: str = "", description: str = "") -> "SolutionMetadata":
        return SolutionMetadata(
            name=self.name or name,
            description=self.description or description,
            extras=dict(self.extras),
        )


class EvaluationResult:
    """Stores the result of evaluating a solution."""

    def __init__(self, valid, score, additional_info):
        self.valid = valid
        self.score = score
        self.additional_info = additional_info


class Solution:
    """Represents a candidate solution in the evolutionary process."""

    def __init__(
        self,
        sol_string,
        metadata: SolutionMetadata | Mapping[str, Any] | None = None,
        evaluation_res: EvaluationResult = None,
    ):
        self.sol_string = sol_string
        self.metadata = SolutionMetadata.coerce(metadata)
        self.evaluation_res = evaluation_res
