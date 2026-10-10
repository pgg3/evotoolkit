"""FunSearch contexts from score-clustered islands."""

from dataclasses import replace

import numpy as np

from evotoolkit.archives import IslandArchive
from evotoolkit.core import ProposalContext, Solution, TaskSpec
from evotoolkit.proposers.templates import INSTRUCTIONS


class FunSearchStrategy:
    name = "funsearch"

    def __init__(self, spec: TaskSpec, *, num_islands: int = 5, num_parents: int = 2, reset_interval: int = 30000):
        if not isinstance(num_parents, int) or num_parents < 1:
            raise ValueError("num_parents must be a positive integer")
        self.spec = spec
        self.config = dict(num_islands=num_islands, num_parents=num_parents, reset_interval=reset_interval)
        self.archive = IslandArchive(num_islands, reset_interval=reset_interval)
        self.instruction = replace(INSTRUCTIONS["funsearch"], num_parents=num_parents)

    def next_batch(self, step: int, max_n: int, remaining_budget: int, rng: np.random.Generator) -> list[ProposalContext]:
        return [
            ProposalContext(
                spec=self.spec,
                parents=self.archive.sample(self.instruction.num_parents, rng=rng),
                instruction=self.instruction,
                best=self.archive.best(),
                hints=[],
                num_proposals=1,
            )
            for _ in range(max(0, min(max_n, remaining_budget)))
        ]

    def observe(self, solutions: list[Solution]) -> None:
        for solution in solutions:
            self.archive.add(solution)

    def should_stop(self, step: int) -> bool:
        return False
