# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for Solution and EvaluationResult data structures."""

from evotoolkit.core import EvaluationResult, Solution, SolutionMetadata


class TestEvaluationResult:
    def test_create_valid(self):
        result = EvaluationResult(valid=True, score=0.95, additional_info={"metric": "mse"})
        assert result.valid is True
        assert result.score == 0.95
        assert result.additional_info == {"metric": "mse"}

    def test_create_invalid(self):
        result = EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "timeout"})
        assert result.valid is False
        assert result.score == float("-inf")
        assert result.additional_info["error"] == "timeout"

    def test_create_empty_additional_info(self):
        result = EvaluationResult(valid=True, score=1.0, additional_info={})
        assert result.additional_info == {}

    def test_score_zero(self):
        result = EvaluationResult(valid=True, score=0.0, additional_info={})
        assert result.score == 0.0

    def test_score_negative(self):
        result = EvaluationResult(valid=True, score=-5.0, additional_info={})
        assert result.score == -5.0


class TestSolution:
    def test_create_minimal(self):
        sol = Solution("def f(x): return x")
        assert sol.sol_string == "def f(x): return x"
        assert sol.metadata == SolutionMetadata()
        assert sol.evaluation_res is None

    def test_create_with_metadata(self):
        sol = Solution("code", metadata={"description": "linear", "score": 1.0})
        assert sol.metadata.description == "linear"
        assert sol.metadata.extras["score"] == 1.0

    def test_create_with_evaluation_res(self):
        res = EvaluationResult(valid=True, score=2.5, additional_info={})
        sol = Solution("code", evaluation_res=res)
        assert sol.evaluation_res is res
        assert sol.evaluation_res.score == 2.5

    def test_create_fully_populated(self):
        res = EvaluationResult(valid=True, score=9.9, additional_info={"runtime": 0.1})
        sol = Solution("code", metadata={"name": "best"}, evaluation_res=res)
        assert sol.sol_string == "code"
        assert sol.metadata.name == "best"
        assert sol.evaluation_res.valid is True

    def test_empty_sol_string(self):
        sol = Solution("")
        assert sol.sol_string == ""

    def test_multiline_sol_string(self):
        code = "def f(x):\n    return x * 2\n"
        sol = Solution(code)
        assert sol.sol_string == code

    def test_evaluation_res_mutation(self):
        sol = Solution("code")
        assert sol.evaluation_res is None
        sol.evaluation_res = EvaluationResult(valid=True, score=1.0, additional_info={})
        assert sol.evaluation_res is not None
        assert sol.evaluation_res.valid is True

    def test_metadata_mutation(self):
        sol = Solution("code", metadata={"key": "value"})
        sol.metadata.extras["key"] = "updated"
        assert sol.metadata.extras["key"] == "updated"
