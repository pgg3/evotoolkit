# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for FunSearchPythonInterface prompt generation and parsing."""

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.python_task.funsearch_interface import FunSearchPythonInterface


def _make_solution(interface) -> Solution:
    return interface.make_solution(
        "def f(x):\n    return x",
    )


@pytest.fixture
def iface(minimal_task):
    return FunSearchPythonInterface(minimal_task)


@pytest.fixture
def sol1():
    return Solution(
        sol_string="def f(x):\n    return x",
        evaluation_res=EvaluationResult(valid=True, score=1.0, additional_info={}),
    )


@pytest.fixture
def sol2():
    return Solution(
        sol_string="def f(x):\n    return x * 2",
        evaluation_res=EvaluationResult(valid=True, score=2.0, additional_info={}),
    )


class TestFunSearchPythonInterfacePrompts:
    def test_get_prompt_single_solution(self, iface, sol1):
        messages = iface.get_prompt([sol1])
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert sol1.sol_string in messages[0]["content"]

    def test_get_prompt_two_solutions(self, iface, sol1, sol2):
        messages = iface.get_prompt([sol1, sol2])
        assert isinstance(messages, list)
        content = messages[0]["content"]
        assert sol1.sol_string in content
        assert sol2.sol_string in content

    def test_get_prompt_empty_fallback(self, iface):
        messages = iface.get_prompt([])
        assert isinstance(messages, list)
        assert len(messages) == 1

    def test_prompt_contains_task_description(self, iface, sol1):
        messages = iface.get_prompt([sol1])
        content = messages[0]["content"]
        assert "Python" in content or "function" in content.lower()

    def test_can_make_solution(self, iface):
        sol = _make_solution(iface)
        assert isinstance(sol, Solution)
        assert len(sol.sol_string) > 0


class TestFunSearchPythonInterfaceParsing:
    def test_parse_python_code_block(self, iface):
        response = "```python\ndef f(x):\n    return x * 3\n```"
        sol = iface.parse_response(response)
        assert "def f" in sol.sol_string

    def test_parse_generic_code_block(self, iface):
        response = "```\ndef g(x):\n    return x + 1\n```"
        sol = iface.parse_response(response)
        assert "def g" in sol.sol_string

    def test_parse_fallback_no_code_block(self, iface):
        response = "def f(x):\n    return x"
        sol = iface.parse_response(response)
        assert isinstance(sol, Solution)
        assert len(sol.sol_string) > 0

    def test_parse_returns_longest_match(self, iface):
        response = "```python\ndef short(): pass\n```\n```python\ndef longer(x):\n    return x * x + x + 1\n```"
        sol = iface.parse_response(response)
        assert "longer" in sol.sol_string

    def test_parse_response_strips_whitespace(self, iface):
        response = "  ```python\ndef f(x):\n    return x\n```  "
        sol = iface.parse_response(response)
        assert not sol.sol_string.startswith(" ")
