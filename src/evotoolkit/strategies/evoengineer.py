"""Engineering search with the best artifact and sampled rationale hints."""

import numpy as np

from evotoolkit.archives import RankedPopulation
from evotoolkit.core import ProposalContext, Solution, TaskSpec
from evotoolkit.proposers.templates import INSTRUCTIONS


class EvoEngineerStrategy:
    name = "evoengineer"

    def __init__(self, spec: TaskSpec, *, pop_size: int = 5):
        if pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        self.spec = spec
        self.config = dict(pop_size=pop_size)
        self.archive = RankedPopulation(pop_size)

    def next_batch(self, step: int, max_n: int, remaining_budget: int, rng: np.random.Generator) -> list[ProposalContext]:
        contexts = []
        count = max(0, min(max_n, remaining_budget))
        descriptions = [solution.rationale for solution in self.archive.solutions if solution.rationale]
        for offset in range(count):
            name = "init" if not len(self.archive) else "mutation" if len(self.archive) == 1 else ("crossover", "mutation")[(step * count + offset) % 2]
            instruction = INSTRUCTIONS[name]
            hints = descriptions.copy() if len(descriptions) <= 3 else [descriptions[index] for index in rng.choice(len(descriptions), size=3, replace=False)]
            contexts.append(
                ProposalContext(
                    spec=self.spec,
                    parents=self.archive.sample(instruction.num_parents, rng=rng),
                    instruction=instruction,
                    best=self.archive.best(),
                    hints=hints,
                    num_proposals=1,
                )
            )
        return contexts

    def observe(self, solutions: list[Solution]) -> None:
        for solution in solutions:
            self.archive.add(solution)

    def should_stop(self, step: int) -> bool:
        return False
