"""Shared data objects for program search."""

import hashlib
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

Files = dict[str, str]


def content_id(files: Files) -> str:
    return hashlib.sha256("\0".join(f"{path}\0{files[path]}" for path in sorted(files)).encode("utf-8")).hexdigest()[:16]


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt: str
    modality: str
    main_file: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        return cls(**{item.name: data[item.name] for item in fields(cls) if item.name in data})


@dataclass
class EvaluationResult:
    valid: bool
    score: float | None
    feedback: str
    metrics: dict[str, Any] = field(default_factory=dict)
    cost_seconds: float = 0.0

    def to_dict(self) -> dict:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        return cls(**{item.name: data[item.name] for item in fields(cls) if item.name in data})


@dataclass(frozen=True)
class Instruction:
    name: str
    num_parents: int
    text: str
    output_format: str


@dataclass
class ProposalContext:
    spec: TaskSpec
    parents: list["Solution"]
    instruction: Instruction
    best: "Solution | None" = None
    hints: list[str] = field(default_factory=list)
    num_proposals: int = 1
    call_id: str = ""  # assigned by Runner; proposers must echo it on every Proposal


@dataclass
class Proposal:
    call_id: str
    files: Files
    name: str = ""
    rationale: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    id: str
    files: Files
    name: str
    rationale: str
    parent_ids: list[str]
    instruction: str
    proposer: str
    call_id: str
    usage: dict[str, Any]
    evaluation: EvaluationResult | None
    created_step: int

    @classmethod
    def create(
        cls,
        *,
        files: Files,
        name: str,
        rationale: str,
        parent_ids: list[str],
        instruction: str,
        proposer: str,
        call_id: str,
        usage: dict[str, Any],
        created_step: int,
        evaluation: EvaluationResult | None = None,
    ) -> "Solution":
        return cls(content_id(files), files, name, rationale, parent_ids, instruction, proposer, call_id, usage, evaluation, created_step)

    @property
    def score(self) -> float | None:
        return self.evaluation.score if self.evaluation is not None and self.evaluation.valid else None

    def to_dict(self) -> dict:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "Solution":
        payload = {item.name: data[item.name] for item in fields(cls) if item.name in data}
        if payload.get("evaluation") is not None:
            payload["evaluation"] = EvaluationResult.from_dict(payload["evaluation"])
        return cls(**payload)
