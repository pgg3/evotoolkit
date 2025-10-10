# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from typing import List

import numpy as np
import scipy

from evotoolkit.core import Solution


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite logits."""
    try:
        if not np.all(np.isfinite(logits)):
            non_finites = set(logits[~np.isfinite(logits)])
            raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
        if not np.issubdtype(logits.dtype, np.floating):
            logits = np.array(logits, dtype=np.float32)

        result = scipy.special.softmax(logits / temperature, axis=-1)
        # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
        index = np.argmax(result)
        result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
        return result
    except TypeError as type_err:
        print(logits)
        raise type_err


class Cluster:
    """A cluster of programs with the same score."""

    def __init__(self, score: float, solution: Solution):
        self.score = score
        self.solutions: List[Solution] = [solution]
        self.lengths: List[int] = [len(solution.sol_string)]

    def register_solution(self, solution: Solution) -> None:
        """Adds solution to the cluster."""
        self.solutions.append(solution)
        self.lengths.append(len(solution.sol_string))

    def sample_solution(self) -> Solution:
        """Samples a solution, giving higher probability to shorter solutions."""
        if len(self.solutions) == 1:
            return self.solutions[0]

        normalized_lengths = (np.array(self.lengths) - min(self.lengths)) / (
            max(self.lengths) + 1e-6
        )
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self.solutions, p=probabilities)


class Island:
    """A sub-population of the programs database."""

    def __init__(
        self,
        solutions_per_prompt: int = 2,
        cluster_sampling_temperature_init: float = 0.1,
        cluster_sampling_temperature_period: int = 30000,
    ):
        self.solutions_per_prompt = solutions_per_prompt
        self.cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self.cluster_sampling_temperature_period = cluster_sampling_temperature_period
        self.clusters: dict[float, Cluster] = {}
        self.num_programs = 0

    def register_solution(self, solution: Solution, score: float) -> None:
        """Stores a solution on this island, in its appropriate cluster."""
        if score not in self.clusters:
            self.clusters[score] = Cluster(score, solution)
        else:
            self.clusters[score].register_solution(solution)
        self.num_programs += 1

    def get_prompt_solutions(self) -> List[Solution]:
        """Constructs a list of solutions for prompt generation."""
        if not self.clusters:
            return []

        # Filter out clusters with -inf scores
        valid_signatures = [
            sig for sig in self.clusters.keys() if not np.isinf(sig) or sig > -np.inf
        ]

        if not valid_signatures:
            return []

        cluster_scores = np.array(
            [self.clusters[sig].score for sig in valid_signatures]
        )

        # Normalize the scores
        max_abs_score = float(np.abs(cluster_scores).max())
        if max_abs_score > 1:
            cluster_scores = cluster_scores.astype(float) / max_abs_score

        # Convert scores to probabilities using softmax with temperature schedule
        period = self.cluster_sampling_temperature_period
        temperature = self.cluster_sampling_temperature_init * (
            1 - (self.num_programs % period) / period
        )
        probabilities = _softmax(cluster_scores, temperature)

        # At the beginning when we have few clusters, place fewer solutions into prompt
        solutions_per_prompt = min(len(valid_signatures), self.solutions_per_prompt)

        idx = np.random.choice(
            len(valid_signatures),
            size=solutions_per_prompt,
            p=probabilities,
            replace=False,
        )
        chosen_signatures = [valid_signatures[i] for i in idx]

        selected_solutions = []
        scores = []
        for signature in chosen_signatures:
            cluster = self.clusters[signature]
            selected_solutions.append(cluster.sample_solution())
            scores.append(cluster.score)

        # Sort solutions by score (ascending, so best solutions come last in prompt)
        indices = np.argsort(scores)
        sorted_solutions = [selected_solutions[i] for i in indices]

        return sorted_solutions

    def get_best_solution(self) -> Solution:
        """Returns the best solution from this island."""
        if not self.clusters:
            return None

        best_score = max(self.clusters.keys())
        best_cluster = self.clusters[best_score]
        return best_cluster.sample_solution()

    def get_best_score(self) -> float:
        """Returns the best score from this island."""
        if not self.clusters:
            return float("-inf")
        return max(self.clusters.keys())
