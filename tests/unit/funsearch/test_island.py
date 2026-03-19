# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for FunSearch Island and Cluster data structures."""

import numpy as np
import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.evo_method.funsearch.island import Cluster, Island, _softmax

# ---------------------------------------------------------------------------
# _softmax tests
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_returns_probabilities_sum_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = _softmax(logits, temperature=1.0)
        assert abs(sum(result) - 1.0) < 1e-10

    def test_higher_logit_gets_higher_probability(self):
        logits = np.array([0.0, 1.0, 2.0])
        result = _softmax(logits, temperature=1.0)
        assert result[2] > result[1] > result[0]

    def test_non_finite_raises_value_error(self):
        logits = np.array([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError, match="non-finite"):
            _softmax(logits, temperature=1.0)

    def test_integer_logits_converted(self):
        logits = np.array([1, 2, 3])
        result = _softmax(logits, temperature=1.0)
        assert result.dtype == np.float32

    def test_single_element(self):
        logits = np.array([5.0])
        result = _softmax(logits, temperature=1.0)
        assert abs(result[0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Cluster tests
# ---------------------------------------------------------------------------


def _make_valid_solution(code: str, score: float) -> Solution:
    return Solution(
        sol_string=code,
        evaluation_res=EvaluationResult(valid=True, score=score, additional_info={}),
    )


class TestCluster:
    def test_init(self):
        sol = _make_valid_solution("def f(): pass", 1.0)
        cluster = Cluster(score=1.0, solution=sol)
        assert cluster.score == 1.0
        assert len(cluster.solutions) == 1
        assert len(cluster.lengths) == 1

    def test_register_solution(self):
        sol1 = _make_valid_solution("x", 1.0)
        sol2 = _make_valid_solution("y = 1", 1.0)
        cluster = Cluster(score=1.0, solution=sol1)
        cluster.register_solution(sol2)
        assert len(cluster.solutions) == 2
        assert len(cluster.lengths) == 2

    def test_sample_solution_single(self):
        sol = _make_valid_solution("x", 1.0)
        cluster = Cluster(score=1.0, solution=sol)
        sampled = cluster.sample_solution()
        assert sampled is sol

    def test_sample_solution_multiple_returns_one(self):
        sol1 = _make_valid_solution("x", 1.0)
        sol2 = _make_valid_solution("y = 1 + 2", 1.0)
        sol3 = _make_valid_solution("z = 'longer string here'", 1.0)
        cluster = Cluster(score=1.0, solution=sol1)
        cluster.register_solution(sol2)
        cluster.register_solution(sol3)
        sampled = cluster.sample_solution()
        assert sampled in [sol1, sol2, sol3]

    def test_lengths_tracked_correctly(self):
        sol1 = _make_valid_solution("ab", 1.0)
        sol2 = _make_valid_solution("abcde", 1.0)
        cluster = Cluster(score=1.0, solution=sol1)
        cluster.register_solution(sol2)
        assert cluster.lengths == [len("ab"), len("abcde")]


# ---------------------------------------------------------------------------
# Island tests
# ---------------------------------------------------------------------------


class TestIsland:
    def test_init_empty(self):
        island = Island()
        assert island.num_programs == 0
        assert len(island.clusters) == 0

    def test_register_solution_creates_cluster(self):
        island = Island()
        sol = _make_valid_solution("def f(): pass", 1.5)
        island.register_solution(sol, score=1.5)
        assert island.num_programs == 1
        assert 1.5 in island.clusters

    def test_register_solution_same_score_adds_to_cluster(self):
        island = Island()
        sol1 = _make_valid_solution("x", 1.0)
        sol2 = _make_valid_solution("y", 1.0)
        island.register_solution(sol1, score=1.0)
        island.register_solution(sol2, score=1.0)
        assert island.num_programs == 2
        assert len(island.clusters[1.0].solutions) == 2

    def test_register_different_scores_creates_multiple_clusters(self):
        island = Island()
        sol1 = _make_valid_solution("x", 1.0)
        sol2 = _make_valid_solution("y", 2.0)
        island.register_solution(sol1, score=1.0)
        island.register_solution(sol2, score=2.0)
        assert len(island.clusters) == 2

    def test_get_prompt_solutions_empty_island(self):
        island = Island()
        result = island.get_prompt_solutions()
        assert result == []

    def test_get_prompt_solutions_returns_list(self):
        island = Island(solutions_per_prompt=2)
        sol1 = _make_valid_solution("x" * 10, 1.0)
        sol2 = _make_valid_solution("y" * 20, 2.0)
        island.register_solution(sol1, score=1.0)
        island.register_solution(sol2, score=2.0)
        result = island.get_prompt_solutions()
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_get_prompt_solutions_single_cluster(self):
        island = Island(solutions_per_prompt=3)
        sol = _make_valid_solution("x", 1.0)
        island.register_solution(sol, score=1.0)
        result = island.get_prompt_solutions()
        assert len(result) == 1

    def test_get_best_solution_empty(self):
        island = Island()
        assert island.get_best_solution() is None

    def test_get_best_solution_returns_highest_score(self):
        island = Island()
        sol1 = _make_valid_solution("low", 0.5)
        sol2 = _make_valid_solution("high", 9.9)
        island.register_solution(sol1, score=0.5)
        island.register_solution(sol2, score=9.9)
        best = island.get_best_solution()
        assert best is not None
        assert best.evaluation_res.score == 9.9

    def test_get_best_score_empty(self):
        island = Island()
        assert island.get_best_score() == float("-inf")

    def test_get_best_score(self):
        island = Island()
        sol = _make_valid_solution("x", 3.7)
        island.register_solution(sol, score=3.7)
        assert island.get_best_score() == 3.7

    def test_prompt_solutions_sorted_ascending(self):
        """Prompt solutions should be sorted ascending (best last)."""
        island = Island(solutions_per_prompt=10)
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, score in enumerate(scores):
            sol = _make_valid_solution(f"code_{i}", score)
            island.register_solution(sol, score=score)
        result = island.get_prompt_solutions()
        result_scores = [s.evaluation_res.score for s in result]
        assert result_scores == sorted(result_scores)
