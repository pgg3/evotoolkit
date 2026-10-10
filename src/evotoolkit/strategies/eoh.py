"""Evolution of Heuristics parent and instruction selection."""

from dataclasses import replace

import numpy as np

from evotoolkit.archives import RankedPopulation
from evotoolkit.core import ProposalContext, Solution, TaskSpec
from evotoolkit.proposers.templates import INSTRUCTIONS


class EoHStrategy:
    name = "eoh"

    def __init__(self, spec: TaskSpec, *, pop_size: int = 5, selection_num: int = 2):
        if not isinstance(selection_num, int) or not 1 <= selection_num <= pop_size:
            raise ValueError("selection_num must be between 1 and pop_size")
        self.spec = spec
        self.config = dict(pop_size=pop_size, selection_num=selection_num)
        self.archive = RankedPopulation(pop_size)
        self.instructions = [
            replace(INSTRUCTIONS[name], num_parents=selection_num) if name in {"e1", "e2"} else INSTRUCTIONS[name] for name in ("e1", "e2", "m1", "m2")
        ]

    def next_batch(self, step: int, max_n: int, remaining_budget: int, rng: np.random.Generator) -> list[ProposalContext]:
        contexts = []
        count = max(0, min(max_n, remaining_budget))
        available = self.instructions if len(self.archive) >= self.instructions[0].num_parents else self.instructions[2:]
        for offset in range(count):
            instruction = INSTRUCTIONS["i1"] if not len(self.archive) else available[(step * count + offset) % len(available)]
            contexts.append(
                ProposalContext(
                    spec=self.spec,
                    parents=self.archive.sample(instruction.num_parents, rng=rng),
                    instruction=instruction,
                    best=self.archive.best(),
                    hints=[],
                    num_proposals=1,
                )
            )
        return contexts

    def observe(self, solutions: list[Solution]) -> None:
        for solution in solutions:
            self.archive.add(solution)

    def should_stop(self, step: int) -> bool:
        return False
