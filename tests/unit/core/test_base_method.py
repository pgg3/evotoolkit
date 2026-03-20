# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for Method static helper methods."""

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.core.base_method import Method


def make_sol(score, valid=True):
    return Solution(
        sol_string=f"code_{score}",
        evaluation_res=EvaluationResult(valid=valid, score=score, additional_info={}),
    )


class TestMethodStaticHelpers:
    def test_get_best_valid_sol_single(self):
        sols = [make_sol(1.0)]
        best = Method._get_best_valid_sol(sols)
        assert best.evaluation_res.score == 1.0

    def test_get_best_valid_sol_picks_max(self):
        sols = [make_sol(1.0), make_sol(3.0), make_sol(2.0)]
        best = Method._get_best_valid_sol(sols)
        assert best.evaluation_res.score == 3.0

    def test_get_best_valid_sol_ignores_invalid(self):
        sols = [make_sol(10.0, valid=False), make_sol(2.0, valid=True), make_sol(1.0, valid=True)]
        best = Method._get_best_valid_sol(sols)
        assert best.evaluation_res.score == 2.0

    def test_get_best_sol_returns_valid(self):
        sols = [make_sol(1.0), make_sol(5.0), make_sol(3.0)]
        best = Method._get_best_sol(sols)
        assert best.evaluation_res.score == 5.0

    def test_get_best_sol_all_invalid_returns_first(self):
        sols = [make_sol(10.0, valid=False), make_sol(20.0, valid=False)]
        assert Method._get_best_sol(sols) is sols[0]

    def test_get_best_valid_sol_no_evaluation_res(self):
        sols_no_eval = [Solution("code_no_eval")]
        # Only solutions without evaluation_res - should raise (max of empty list)
        with pytest.raises((ValueError, Exception)):
            Method._get_best_valid_sol(sols_no_eval)

    def test_get_best_valid_sol_negative_scores(self):
        sols = [make_sol(-1.0), make_sol(-3.0), make_sol(-2.0)]
        best = Method._get_best_valid_sol(sols)
        assert best.evaluation_res.score == -1.0
