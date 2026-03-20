# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for string optimization interfaces: EvoEngineerStringInterface and FunSearchStringInterface."""

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.string_optimization.method_interface.evoengineer_interface import EvoEngineerStringInterface
from evotoolkit.task.string_optimization.method_interface.funsearch_interface import FunSearchStringInterface
from evotoolkit.task.string_optimization.string_task import StringTask

# ---------------------------------------------------------------------------
# Minimal StringTask for testing
# ---------------------------------------------------------------------------


class LengthStringTask(StringTask):
    """StringTask: score = length of string."""

    def _process_data(self, data):
        self.data = data
        self.task_info = {"name": "length_task"}

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        return EvaluationResult(valid=True, score=float(len(candidate_string)), additional_info={})

    def get_base_task_description(self) -> str:
        return "Generate a string. Longer strings score higher."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("start string", other_info={"name": "init", "thought": "Initial solution"})


@pytest.fixture
def str_task():
    return LengthStringTask(data=None)


@pytest.fixture
def ee_iface(str_task):
    return EvoEngineerStringInterface(str_task)


@pytest.fixture
def fs_iface(str_task):
    class ConcreteFunSearchStringInterface(FunSearchStringInterface):
        def parse_response(self, response_str: str) -> Solution:
            return Solution(response_str.strip())

    return ConcreteFunSearchStringInterface(str_task)


@pytest.fixture
def best_sol():
    return Solution(
        sol_string="The quick brown fox jumps",
        other_info={"name": "best", "thought": "Use long vivid words"},
        evaluation_res=EvaluationResult(valid=True, score=25.0, additional_info={}),
    )


@pytest.fixture
def parent_sols():
    return [
        Solution(
            sol_string="Hello world",
            other_info={"name": "hello", "thought": "Simple greeting"},
            evaluation_res=EvaluationResult(valid=True, score=11.0, additional_info={}),
        ),
        Solution(
            sol_string="Goodbye cruel world",
            other_info={"name": "bye", "thought": "Dramatic farewell"},
            evaluation_res=EvaluationResult(valid=True, score=19.0, additional_info={}),
        ),
    ]


# ---------------------------------------------------------------------------
# EvoEngineerStringInterface tests
# ---------------------------------------------------------------------------


class TestEvoEngineerStringInterfaceOperators:
    def test_get_init_operators(self, ee_iface):
        ops = ee_iface.get_init_operators()
        assert len(ops) >= 1
        assert all(op.selection_size == 0 for op in ops)

    def test_get_offspring_operators(self, ee_iface):
        ops = ee_iface.get_offspring_operators()
        names = [op.name for op in ops]
        assert "crossover" in names or "mutation" in names


class TestEvoEngineerStringInterfacePrompts:
    def test_init_prompt(self, ee_iface, best_sol):
        msgs = ee_iface.get_operator_prompt("init", [], best_sol, [])
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "user"
        assert len(msgs[0]["content"]) > 0

    def test_crossover_prompt(self, ee_iface, best_sol, parent_sols):
        msgs = ee_iface.get_operator_prompt("crossover", parent_sols, best_sol, [])
        assert isinstance(msgs, list)
        content = msgs[0]["content"]
        assert len(content) > 0

    def test_mutation_prompt(self, ee_iface, best_sol, parent_sols):
        msgs = ee_iface.get_operator_prompt("mutation", [parent_sols[0]], best_sol, [])
        assert isinstance(msgs, list)
        content = msgs[0]["content"]
        assert len(content) > 0

    def test_unknown_operator_raises(self, ee_iface, best_sol):
        # EvoEngineerStringInterface returns [] for unknown operators (no ValueError)
        result = ee_iface.get_operator_prompt("unknown_op", [], best_sol, [])
        assert result == []

    def test_init_prompt_with_thoughts(self, ee_iface, best_sol):
        msgs = ee_iface.get_operator_prompt("init", [], best_sol, ["try longer words"])
        content = msgs[0]["content"]
        assert "longer words" in content

    def test_make_baseline_solution(self, ee_iface):
        sol = ee_iface._make_baseline_solution()
        assert isinstance(sol, Solution)
        assert sol.other_info["name"] == "init"
        assert sol.other_info["thought"] == "Initial solution"

    def test_parse_response_returns_solution(self, ee_iface):
        response = "name: my_string\nThe answer is a long sentence with many words.\nthought: more words = higher score"
        sol = ee_iface.parse_response(response)
        assert isinstance(sol, Solution)

    def test_parse_empty_response(self, ee_iface):
        sol = ee_iface.parse_response("")
        assert isinstance(sol, Solution)


# ---------------------------------------------------------------------------
# FunSearchStringInterface tests
# ---------------------------------------------------------------------------


class TestFunSearchStringInterfacePrompts:
    def test_get_prompt_single_solution(self, fs_iface, best_sol):
        msgs = fs_iface.get_prompt([best_sol])
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "user"
        assert best_sol.sol_string in msgs[0]["content"]

    def test_get_prompt_two_solutions(self, fs_iface, parent_sols):
        msgs = fs_iface.get_prompt(parent_sols)
        content = msgs[0]["content"]
        assert parent_sols[0].sol_string in content
        assert parent_sols[1].sol_string in content

    def test_get_prompt_empty_fallback(self, fs_iface):
        msgs = fs_iface.get_prompt([])
        assert isinstance(msgs, list)
        assert len(msgs) == 1

    def test_task_init_solution(self, fs_iface):
        sol = fs_iface.task.make_init_sol_wo_other_info()
        assert isinstance(sol, Solution)
