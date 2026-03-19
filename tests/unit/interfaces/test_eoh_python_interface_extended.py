# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EoHPythonInterface — covering uncovered prompt/parse methods."""

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.python_task.method_interface.eoh_interface import EoHPythonInterface


def _make_algo_solution(code: str, score: float, algorithm: str = "test algo") -> Solution:
    return Solution(
        sol_string=code,
        other_info={"algorithm": algorithm},
        evaluation_res=EvaluationResult(valid=True, score=score, additional_info={}),
    )


def _make_no_algo_solution(code: str, score: float) -> Solution:
    return Solution(
        sol_string=code,
        other_info={},
        evaluation_res=EvaluationResult(valid=True, score=score, additional_info={}),
    )


class TestEoHPythonInterfacePrompts:
    def test_get_prompt_i1_returns_messages(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        prompt = iface.get_prompt_i1()
        assert isinstance(prompt, list)
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert len(prompt[0]["content"]) > 0

    def test_get_prompt_e1_with_algorithm_desc(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        individuals = [
            _make_algo_solution("def f(x): return x", 1.0, "linear"),
            _make_algo_solution("def f(x): return x*2", 2.0, "double"),
        ]
        prompt = iface.get_prompt_e1(individuals)
        assert len(prompt) == 1
        content = prompt[0]["content"]
        assert "linear" in content
        assert "double" in content

    def test_get_prompt_e1_without_algorithm_desc(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        individuals = [
            _make_no_algo_solution("def f(x): return x", 1.0),
        ]
        prompt = iface.get_prompt_e1(individuals)
        content = prompt[0]["content"]
        # Should fallback to "Python Code 1"
        assert "Python Code 1" in content

    def test_get_prompt_e2_with_algorithm_desc(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        individuals = [
            _make_algo_solution("def f(x): return x", 1.0, "linear"),
            _make_algo_solution("def f(x): return x**2", 2.0, "quadratic"),
        ]
        prompt = iface.get_prompt_e2(individuals)
        assert len(prompt) == 1
        content = prompt[0]["content"]
        assert "backbone" in content.lower()

    def test_get_prompt_e2_without_algorithm_desc(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        individuals = [_make_no_algo_solution("def f(x): return x", 1.0)]
        prompt = iface.get_prompt_e2(individuals)
        content = prompt[0]["content"]
        assert "Python code 1" in content

    def test_get_prompt_m1_with_algorithm(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        individual = _make_algo_solution("def f(x): return x", 1.0, "my_algo")
        prompt = iface.get_prompt_m1(individual)
        content = prompt[0]["content"]
        assert "my_algo" in content

    def test_get_prompt_m1_without_algorithm(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        individual = _make_no_algo_solution("def f(x): return x", 1.0)
        prompt = iface.get_prompt_m1(individual)
        content = prompt[0]["content"]
        assert "Current algorithm" in content

    def test_get_prompt_m2_with_algorithm(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        individual = _make_algo_solution("def f(x): return x + 1", 1.0, "shift_algo")
        prompt = iface.get_prompt_m2(individual)
        content = prompt[0]["content"]
        assert "shift_algo" in content
        assert "parameter" in content.lower()

    def test_get_prompt_m2_without_algorithm(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        individual = _make_no_algo_solution("def f(x): return x", 1.0)
        prompt = iface.get_prompt_m2(individual)
        content = prompt[0]["content"]
        assert "Current algorithm" in content


class TestEoHPythonInterfaceParseResponse:
    def test_parse_response_with_python_code_block(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        response = "{linear function}\n```python\ndef f(x):\n    return x * 2\n```"
        sol = iface.parse_response(response)
        assert "def f(x)" in sol.sol_string
        assert sol.other_info["algorithm"] is not None

    def test_parse_response_uppercase_python_block(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        response = "```Python\ndef f(x):\n    return x + 1\n```"
        sol = iface.parse_response(response)
        assert "def f(x)" in sol.sol_string

    def test_parse_response_generic_code_block(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        response = "```\ndef f(x):\n    return 42\n```"
        sol = iface.parse_response(response)
        assert "42" in sol.sol_string

    def test_parse_response_no_code_block(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        response = "{test algo}\ndef f(x): return x"
        sol = iface.parse_response(response)
        # Falls back to raw content without algorithm part
        assert sol is not None

    def test_parse_response_no_algorithm_desc(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        response = "```python\ndef f(x):\n    return x\n```"
        sol = iface.parse_response(response)
        assert sol.other_info["algorithm"] is None

    def test_parse_response_exception_in_algorithm_extraction(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        # Empty response
        sol = iface.parse_response("")
        assert sol is not None

    def test_parse_response_multiple_code_blocks_picks_longest(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        response = "```python\ndef f(x):\n    return x\n```\n```python\ndef f(x):\n    # longer implementation\n    val = x * 2\n    return val\n```"
        sol = iface.parse_response(response)
        assert "longer" in sol.sol_string or "val" in sol.sol_string
