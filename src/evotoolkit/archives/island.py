"""Score-clustered islands whose placement and resets can be replayed."""

import numpy as np

from evotoolkit.core import Solution

from .ranked import is_rankable


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    weights = np.exp((values - values.max()) / temperature)
    weights = np.maximum(weights, np.finfo(float).tiny)
    return weights / weights.sum()


class IslandArchive:
    """Hash-route new solutions; reset weak islands after accepted solutions.

    Unlike time-based resets, these rules reconstruct exactly from the log.
    The first valid solution seeds every island. Each island clusters by score
    and favors shorter artifacts within a cluster, as in FunSearch.
    """

    def __init__(
        self,
        num_islands: int = 5,
        *,
        reset_interval: int = 30000,
        cluster_sampling_temperature_init: float = 0.1,
        cluster_sampling_temperature_period: int = 30000,
    ):
        for name, value in (
            ("num_islands", num_islands),
            ("reset_interval", reset_interval),
            ("cluster_sampling_temperature_period", cluster_sampling_temperature_period),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(cluster_sampling_temperature_init) or cluster_sampling_temperature_init <= 0:
            raise ValueError("cluster_sampling_temperature_init must be finite and positive")
        self.num_islands = num_islands
        self.reset_interval = reset_interval
        self.temperature = cluster_sampling_temperature_init
        self.temperature_period = cluster_sampling_temperature_period
        self.islands: list[dict[float, dict[str, Solution]]] = [{} for _ in range(num_islands)]
        self._seen: set[str] = set()
        self._best: Solution | None = None

    def add(self, solution: Solution) -> None:
        if not is_rankable(solution) or solution.id in self._seen:
            return
        targets = range(self.num_islands) if not self._seen else [int(solution.id, 16) % self.num_islands]
        for index in targets:
            self.islands[index].setdefault(solution.score, {})[solution.id] = solution
        self._seen.add(solution.id)
        if self._best is None or (-solution.score, solution.id) < (-self._best.score, self._best.id):
            self._best = solution
        if len(self._seen) % self.reset_interval == 0:
            self._reset()

    def _reset(self) -> None:
        ordered = sorted(range(self.num_islands), key=lambda index: (max(self.islands[index]), index))
        count = self.num_islands // 2
        for weak, strong in zip(ordered[:count], reversed(ordered[count:])):
            cluster = self.islands[strong][max(self.islands[strong])]
            founder = min(cluster.values(), key=lambda solution: solution.id)
            self.islands[weak] = {founder.score: {founder.id: founder}}

    def sample(self, n: int, *, rng: np.random.Generator) -> list[Solution]:
        if n <= 0 or not self._seen:
            return []
        island = self.islands[int(rng.integers(self.num_islands))]
        scores = np.array(sorted(island))
        normalized_scores = scores / max(1.0, float(np.abs(scores).max()))
        count = sum(len(cluster) for cluster in island.values())
        temperature = self.temperature * (1 - (count % self.temperature_period) / self.temperature_period)
        indices = rng.choice(len(scores), size=min(n, len(scores)), replace=False, p=_softmax(normalized_scores, temperature))
        selected = []
        for index in sorted(indices):
            solutions = sorted(island[scores[index]].values(), key=lambda solution: solution.id)
            lengths = np.array([sum(len(content) for content in solution.files.values()) for solution in solutions])
            normalized_lengths = (lengths - lengths.min()) / (lengths.max() + 1e-6)
            chosen = int(rng.choice(len(solutions), p=_softmax(-normalized_lengths, 1.0)))
            selected.append(solutions[chosen])
        return selected

    def best(self) -> Solution | None:
        return self._best

    def __len__(self) -> int:
        return len({identifier for island in self.islands for cluster in island.values() for identifier in cluster})
