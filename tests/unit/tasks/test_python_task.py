# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for PythonTask base class behavior."""

from evotoolkit.core import EvaluationResult, Solution


class TestMinimalPythonTask:
    def test_task_type_is_python(self, minimal_task):
        assert minimal_task.get_task_type() == "Python"

    def test_task_info_populated(self, minimal_task):
        info = minimal_task.get_task_info()
        assert isinstance(info, dict)
        assert "name" in info

    def test_description_is_string(self, minimal_task):
        desc = minimal_task.get_base_task_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_init_sol_is_solution(self, minimal_task):
        sol = minimal_task.make_init_sol_wo_other_info()
        assert isinstance(sol, Solution)
        assert isinstance(sol.sol_string, str)


class TestPythonTaskEvaluateCode:
    def test_valid_code(self, minimal_task):
        code = "def f(x):\n    return x * 2"
        result = minimal_task.evaluate_code(code)
        assert isinstance(result, EvaluationResult)
        assert result.valid is True
        assert result.score == 2.0  # f(1) = 2

    def test_syntax_error_returns_invalid(self, minimal_task):
        code = "def f(x !! return x"
        result = minimal_task.evaluate_code(code)
        assert result.valid is False
        assert result.score == float("-inf")

    def test_runtime_error_returns_invalid(self, minimal_task):
        code = "def f(x):\n    raise ValueError('boom')"
        result = minimal_task.evaluate_code(code)
        assert result.valid is False

    def test_missing_function_returns_invalid(self, minimal_task):
        code = "x = 1"
        result = minimal_task.evaluate_code(code)
        assert result.valid is False

    def test_empty_code_returns_invalid(self, minimal_task):
        result = minimal_task.evaluate_code("")
        assert result.valid is False


class TestPythonTaskEvaluateSolution:
    def test_evaluate_solution_delegates_to_evaluate_code(self, minimal_task):
        sol = Solution("def f(x):\n    return x + 1")
        result = minimal_task.evaluate_solution(sol)
        assert result.valid is True
        assert result.score == 2.0  # f(1) = 2

    def test_evaluate_solution_invalid_code(self, minimal_task):
        sol = Solution("not valid python !!!")
        result = minimal_task.evaluate_solution(sol)
        assert result.valid is False
