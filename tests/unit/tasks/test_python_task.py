# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for PythonTask base class behavior."""

from evotoolkit.core import EvaluationResult, Solution


class TestMinimalPythonTask:
    def test_spec_has_python_modality(self, minimal_task):
        assert minimal_task.spec.modality == "python"

    def test_spec_contains_name_and_prompt(self, minimal_task):
        assert minimal_task.spec.name == "minimal"
        assert isinstance(minimal_task.spec.prompt, str)
        assert len(minimal_task.spec.prompt) > 0

    def test_spec_can_carry_initial_solution(self, minimal_task):
        assert minimal_task.spec.initial_solution.startswith("def f")


class TestPythonTaskEvaluate:
    def test_valid_code(self, minimal_task):
        code = "def f(x):\n    return x * 2"
        result = minimal_task.evaluate(Solution(code))
        assert isinstance(result, EvaluationResult)
        assert result.valid is True
        assert result.score == 2.0

    def test_syntax_error_returns_invalid(self, minimal_task):
        code = "def f(x !! return x"
        result = minimal_task.evaluate(Solution(code))
        assert result.valid is False
        assert result.score == float("-inf")

    def test_runtime_error_returns_invalid(self, minimal_task):
        code = "def f(x):\n    raise ValueError('boom')"
        result = minimal_task.evaluate(Solution(code))
        assert result.valid is False

    def test_missing_function_returns_invalid(self, minimal_task):
        code = "x = 1"
        result = minimal_task.evaluate(Solution(code))
        assert result.valid is False

    def test_empty_code_returns_invalid(self, minimal_task):
        result = minimal_task.evaluate(Solution(""))
        assert result.valid is False

    def test_evaluate_solution_uses_sol_string(self, minimal_task):
        sol = Solution("def f(x):\n    return x + 1")
        result = minimal_task.evaluate(sol)
        assert result.valid is True
        assert result.score == 2.0
