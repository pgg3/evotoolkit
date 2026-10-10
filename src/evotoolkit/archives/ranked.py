"""A bounded population with the original rank-based parent selection."""

import math

import numpy as np

from evotoolkit.core import Solution


def is_rankable(solution: Solution) -> bool:
    evaluation = solution.evaluation
    return bool(evaluation and evaluation.valid and evaluation.score is not None and math.isfinite(evaluation.score))


class RankedPopulation:
    def __init__(self, max_size: int = 5):
        if not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size must be a positive integer")
        self.max_size = max_size
        self._solutions: dict[str, Solution] = {}

    @property
    def solutions(self) -> tuple[Solution, ...]:
        return tuple(self._solutions.values())

    def add(self, solution: Solution) -> None:
        if not is_rankable(solution) or solution.id in self._solutions:
            return
        self._solutions[solution.id] = solution
        ranked = sorted(self._solutions.values(), key=lambda item: (-item.score, item.id))
        self._solutions = {item.id: item for item in ranked[: self.max_size]}

    def sample(self, n: int, *, rng: np.random.Generator) -> list[Solution]:
        if n <= 0 or not self._solutions:
            return []
        ranked = self.solutions
        weights = 1 / (np.arange(len(ranked)) + len(ranked))
        indices = rng.choice(len(ranked), size=min(n, len(ranked)), p=weights / weights.sum())
        return [ranked[index] for index in indices]

    def best(self) -> Solution | None:
        return next(iter(self._solutions.values()), None)

    def __len__(self) -> int:
        return len(self._solutions)
