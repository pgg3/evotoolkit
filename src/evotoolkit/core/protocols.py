"""Structural contracts; implementations do not need to inherit these types."""

from typing import Protocol

import numpy as np

from .types import EvaluationResult, Files, Proposal, ProposalContext, Solution, TaskSpec


class Evaluator(Protocol):
    spec: TaskSpec

    def evaluate(self, files: Files) -> EvaluationResult: ...


class Proposer(Protocol):
    name: str

    def propose(self, ctx: ProposalContext) -> list[Proposal]: ...


class Archive(Protocol):
    def add(self, solution: Solution) -> None: ...
    def sample(self, n: int, *, rng: np.random.Generator) -> list[Solution]: ...
    def best(self) -> Solution | None: ...
    def __len__(self) -> int: ...


class Strategy(Protocol):
    name: str
    archive: Archive
    config: dict  # hyperparameters; logged at run_start and checked on resume

    def next_batch(self, step: int, max_n: int, remaining_budget: int, rng: np.random.Generator) -> list[ProposalContext]: ...
    def observe(self, solutions: list[Solution]) -> None: ...
    def should_stop(self, step: int) -> bool: ...
