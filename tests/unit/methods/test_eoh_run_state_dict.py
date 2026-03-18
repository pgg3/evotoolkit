# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EoHRunStateDict serialization."""

import os

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.evo_method.eoh.run_state_dict import EoHRunStateDict


def make_sol(code, score=1.0, valid=True):
    return Solution(
        sol_string=code,
        other_info={"algorithm": "test"},
        evaluation_res=EvaluationResult(valid=valid, score=score, additional_info={}),
    )


@pytest.fixture
def empty_state():
    return EoHRunStateDict(task_info={"name": "test_task"})


@pytest.fixture
def populated_state():
    state = EoHRunStateDict(
        task_info={"name": "test_task"},
        generation=3,
        tot_sample_nums=15,
        population=[make_sol("def f(): return 1", 1.0), make_sol("def f(): return 2", 2.0)],
    )
    state.sol_history = [make_sol("def f(): return 0", 0.5)]
    return state


class TestEoHRunStateDictInit:
    def test_default_values(self, empty_state):
        assert empty_state.generation == 0
        assert empty_state.tot_sample_nums == 0
        assert empty_state.is_done is False
        assert empty_state.sol_history == []
        assert empty_state.population == []

    def test_task_info_stored(self, empty_state):
        assert empty_state.task_info == {"name": "test_task"}


class TestEoHRunStateDictSerialization:
    def test_to_json_contains_keys(self, empty_state):
        data = empty_state.to_json()
        assert "generation" in data
        assert "tot_sample_nums" in data
        assert "population" in data
        assert "is_done" in data
        assert "task_info" in data

    def test_roundtrip_empty(self, empty_state):
        data = empty_state.to_json()
        restored = EoHRunStateDict.from_json(data)
        assert restored.generation == 0
        assert restored.population == []
        assert restored.is_done is False

    def test_roundtrip_with_population(self, populated_state):
        data = populated_state.to_json()
        restored = EoHRunStateDict.from_json(data)
        assert restored.generation == 3
        assert len(restored.population) == 2
        assert restored.population[0].sol_string == "def f(): return 1"
        assert restored.population[1].evaluation_res.score == 2.0

    def test_roundtrip_to_file(self, populated_state, tmp_path):
        file_path = str(tmp_path / "run_state.json")
        populated_state.to_json_file(file_path)
        assert os.path.exists(file_path)
        restored = EoHRunStateDict.from_json_file(file_path)
        assert restored.generation == populated_state.generation
        assert len(restored.population) == len(populated_state.population)

    def test_solution_without_eval_serialized(self, empty_state):
        sol = Solution("unevaluated", other_info={"algorithm": None})
        empty_state.population = [sol]
        data = empty_state.to_json()
        assert data["population"][0]["evaluation_res"] is None
