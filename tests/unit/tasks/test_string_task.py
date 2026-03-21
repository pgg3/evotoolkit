# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for StringTask base class."""

import pytest

from evotoolkit.core import EvaluationResult, Solution, TaskSpec
from evotoolkit.task.string_optimization.string_task import StringTask


class SimpleStringTask(StringTask):
    """Concrete StringTask for testing: scores by string length."""

    def build_string_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="simple_string",
            prompt="Produce a string. Longer is better.",
            modality="string",
        )

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        return EvaluationResult(
            valid=True,
            score=float(len(candidate_string)),
            additional_info={"length": len(candidate_string)},
        )


class ErroringStringTask(StringTask):
    """StringTask that always raises an exception during evaluation."""

    def build_string_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="erroring_string",
            prompt="This task always errors.",
            modality="string",
        )

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        raise ValueError("Always fails")


@pytest.fixture
def simple_task():
    return SimpleStringTask(data=None)


@pytest.fixture
def erroring_task():
    return ErroringStringTask(data=None)


class TestStringTaskBasics:
    def test_spec_has_string_modality(self, simple_task):
        assert simple_task.spec.modality == "string"

    def test_evaluate_returns_valid(self, simple_task):
        result = simple_task.evaluate(Solution("hello world"))
        assert result.valid is True
        assert result.score == 11.0

    def test_evaluate_empty_string(self, simple_task):
        result = simple_task.evaluate(Solution(""))
        assert result.valid is True
        assert result.score == 0.0

    def test_longer_scores_higher(self, simple_task):
        short_res = simple_task.evaluate(Solution("hi"))
        long_res = simple_task.evaluate(Solution("hello world!"))
        assert long_res.score > short_res.score

    def test_evaluate_exception_returns_invalid(self, erroring_task):
        result = erroring_task.evaluate(Solution("anything"))
        assert result.valid is False
        assert result.score == float("-inf")
        assert "error" in result.additional_info

    def test_spec_does_not_define_task_seed(self, simple_task):
        assert not hasattr(simple_task.spec, "initial_solution")

    def test_task_prompt(self, simple_task):
        assert isinstance(simple_task.spec.prompt, str)
        assert len(simple_task.spec.prompt) > 0
