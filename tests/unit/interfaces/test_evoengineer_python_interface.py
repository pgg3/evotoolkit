# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EvoEngineerPythonInterface prompt generation and parsing."""

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.python_task.evoengineer_interface import EvoEngineerPythonInterface


@pytest.fixture
def iface(minimal_task):
    return EvoEngineerPythonInterface(minimal_task)


@pytest.fixture
def best_sol():
    return Solution(
        sol_string="def f(x):\n    return x",
        metadata={"name": "linear", "description": "Simple linear function"},
        evaluation_res=EvaluationResult(valid=True, score=1.0, additional_info={}),
    )


@pytest.fixture
def parent_sols():
    return [
        Solution(
            sol_string="def f(x):\n    return x + 1",
            metadata={"name": "plus_one", "description": "Add one"},
            evaluation_res=EvaluationResult(valid=True, score=2.0, additional_info={}),
        ),
        Solution(
            sol_string="def f(x):\n    return x * 2",
            metadata={"name": "double", "description": "Double it"},
            evaluation_res=EvaluationResult(valid=True, score=3.0, additional_info={}),
        ),
    ]


class TestEvoEngineerPythonInterfaceOperators:
    def test_get_init_operators(self, iface):
        ops = iface.get_init_operators()
        assert len(ops) >= 1
        assert ops[0].selection_size == 0

    def test_get_offspring_operators(self, iface):
        ops = iface.get_offspring_operators()
        names = [op.name for op in ops]
        assert "crossover" in names or "mutation" in names


class TestEvoEngineerPythonInterfacePrompts:
    def test_init_operator_prompt(self, iface, best_sol):
        messages = iface.get_operator_prompt("init", [], best_sol, [])
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_crossover_operator_prompt(self, iface, best_sol, parent_sols):
        messages = iface.get_operator_prompt("crossover", parent_sols, best_sol, [])
        assert isinstance(messages, list)
        content = messages[0]["content"]
        assert "crossover" in content.lower() or "combine" in content.lower() or "parent" in content.lower()

    def test_mutation_operator_prompt(self, iface, best_sol, parent_sols):
        messages = iface.get_operator_prompt("mutation", [parent_sols[0]], best_sol, [])
        assert isinstance(messages, list)
        content = messages[0]["content"]
        assert len(content) > 0

    def test_unknown_operator_raises(self, iface, best_sol):
        with pytest.raises(ValueError, match="Unknown operator"):
            iface.get_operator_prompt("nonexistent_op", [], best_sol, [])

    def test_init_prompt_with_thoughts(self, iface, best_sol):
        messages = iface.get_operator_prompt("init", [], best_sol, ["thought 1", "thought 2"])
        content = messages[0]["content"]
        assert "thought 1" in content

    def test_crossover_prompt_with_thoughts(self, iface, best_sol, parent_sols):
        messages = iface.get_operator_prompt("crossover", parent_sols, best_sol, ["optimization hint"])
        content = messages[0]["content"]
        assert "optimization hint" in content

    def test_prompt_contains_task_description(self, iface, best_sol):
        messages = iface.get_operator_prompt("init", [], best_sol, [])
        content = messages[0]["content"]
        # Task description should appear in prompt
        assert "Python" in content or "function" in content.lower()


class TestEvoEngineerPythonInterfaceParsing:
    def test_parse_python_code_block(self, iface):
        response = "name: my_func\ncode:\n```python\ndef f(x):\n    return x * 2\n```\nthought: doubled it"
        sol = iface.parse_response(response)
        assert isinstance(sol, Solution)

    def test_parse_empty_response(self, iface):
        sol = iface.parse_response("")
        assert isinstance(sol, Solution)
        assert sol.sol_string == ""

    def test_parse_fallback_code_block(self, iface):
        response = "```python\ndef f(x):\n    return x + 99\n```"
        sol = iface.parse_response(response)
        assert isinstance(sol, Solution)
        assert "f" in sol.sol_string or len(sol.sol_string) > 0

    def test_parse_returns_solution_always(self, iface):
        # Even garbage input should return a Solution (not raise)
        sol = iface.parse_response("this is completely unparseable garbage!!!")
        assert isinstance(sol, Solution)

    def test_parse_stores_metadata(self, iface):
        response = "name: my_impl\ncode:\n```python\ndef f(x): return x\n```\nthought: simple"
        sol = iface.parse_response(response)
        assert sol.metadata.name == "my_impl"
        assert sol.metadata.description == "simple"
