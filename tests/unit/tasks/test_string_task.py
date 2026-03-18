# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for StringTask base class."""

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.string_optimization.string_task import StringTask


class SimpleStringTask(StringTask):
    """Concrete StringTask for testing: scores by string length."""

    def _process_data(self, data):
        self.data = data
        self.task_info = {"name": "simple_string"}

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        return EvaluationResult(
            valid=True,
            score=float(len(candidate_string)),
            additional_info={"length": len(candidate_string)},
        )

    def get_base_task_description(self) -> str:
        return "Produce a string. Longer is better."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("hello")


class ErroringStringTask(StringTask):
    """StringTask that always raises an exception during evaluation."""

    def _process_data(self, data):
        self.data = data
        self.task_info = {}

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        raise ValueError("Always fails")

    def get_base_task_description(self) -> str:
        return "This task always errors."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("start")


@pytest.fixture
def simple_task():
    return SimpleStringTask(data=None)


@pytest.fixture
def erroring_task():
    return ErroringStringTask(data=None)


class TestStringTaskBasics:
    def test_get_task_type(self, simple_task):
        assert simple_task.get_task_type() == "String"

    def test_evaluate_code_returns_valid(self, simple_task):
        result = simple_task.evaluate_code("hello world")
        assert result.valid is True
        assert result.score == 11.0

    def test_evaluate_code_empty_string(self, simple_task):
        result = simple_task.evaluate_code("")
        assert result.valid is True
        assert result.score == 0.0

    def test_evaluate_code_longer_scores_higher(self, simple_task):
        short_res = simple_task.evaluate_code("hi")
        long_res = simple_task.evaluate_code("hello world!")
        assert long_res.score > short_res.score

    def test_evaluate_code_exception_returns_invalid(self, erroring_task):
        result = erroring_task.evaluate_code("anything")
        assert result.valid is False
        assert result.score == float("-inf")
        assert "error" in result.additional_info

    def test_evaluate_solution_delegates(self, simple_task):
        sol = Solution("test string")
        result = simple_task.evaluate_solution(sol)
        assert result.valid is True
        assert result.score == 11.0

    def test_make_init_sol(self, simple_task):
        sol = simple_task.make_init_sol_wo_other_info()
        assert isinstance(sol, Solution)
        assert sol.sol_string == "hello"

    def test_task_description(self, simple_task):
        desc = simple_task.get_base_task_description()
        assert isinstance(desc, str)
        assert len(desc) > 0
