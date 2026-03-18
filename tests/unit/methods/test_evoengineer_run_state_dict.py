# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EvoEngineerRunStateDict serialization."""

import os

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.evo_method.evoengineer.run_state_dict import EvoEngineerRunStateDict


def make_sol(code, score=1.0, valid=True):
    return Solution(
        sol_string=code,
        other_info={"name": "test", "thought": "test thought"},
        evaluation_res=EvaluationResult(valid=valid, score=score, additional_info={}),
    )


class TestEvoEngineerRunStateDictInit:
    def test_default_values(self):
        state = EvoEngineerRunStateDict(task_info={"name": "t"})
        assert state.generation == 0
        assert state.tot_sample_nums == 0
        assert state.is_done is False
        assert state.sol_history == []
        assert state.population == []
        assert state.usage_history == {}

    def test_task_info_stored(self):
        state = EvoEngineerRunStateDict(task_info={"key": "val"})
        assert state.task_info == {"key": "val"}

    def test_custom_values(self):
        pop = [make_sol("def f(): return 1")]
        state = EvoEngineerRunStateDict(
            task_info={},
            generation=5,
            tot_sample_nums=20,
            population=pop,
            is_done=True,
        )
        assert state.generation == 5
        assert state.tot_sample_nums == 20
        assert state.is_done is True
        assert len(state.population) == 1


class TestEvoEngineerRunStateDictSerialization:
    def test_to_json_contains_keys(self):
        state = EvoEngineerRunStateDict(task_info={"name": "t"})
        data = state.to_json()
        for key in ("task_info", "generation", "tot_sample_nums", "population", "is_done", "current_best"):
            assert key in data

    def test_roundtrip_empty(self):
        state = EvoEngineerRunStateDict(task_info={"name": "t"})
        data = state.to_json()
        restored = EvoEngineerRunStateDict.from_json(data)
        assert restored.generation == 0
        assert restored.population == []
        assert restored.is_done is False

    def test_roundtrip_with_population(self):
        sol1 = make_sol("def f(): return 1", 1.0)
        sol2 = make_sol("def f(): return 2", 2.0)
        state = EvoEngineerRunStateDict(
            task_info={"name": "t"},
            generation=3,
            tot_sample_nums=10,
            population=[sol1, sol2],
        )
        data = state.to_json()
        restored = EvoEngineerRunStateDict.from_json(data)
        assert restored.generation == 3
        assert len(restored.population) == 2
        assert restored.population[0].sol_string == "def f(): return 1"
        assert restored.population[1].evaluation_res.score == 2.0

    def test_roundtrip_to_file(self, tmp_path):
        state = EvoEngineerRunStateDict(
            task_info={"name": "t"},
            generation=2,
            population=[make_sol("def f(): return 0")],
        )
        file_path = str(tmp_path / "state.json")
        state.to_json_file(file_path)
        assert os.path.exists(file_path)
        restored = EvoEngineerRunStateDict.from_json_file(file_path)
        assert restored.generation == 2
        assert len(restored.population) == 1

    def test_current_best_none_when_empty(self):
        state = EvoEngineerRunStateDict(task_info={})
        data = state.to_json()
        assert data["current_best"] is None

    def test_current_best_from_sol_history(self):
        state = EvoEngineerRunStateDict(task_info={})
        state.sol_history = [make_sol("def f(): return 5", 5.0), make_sol("def f(): return 1", 1.0)]
        data = state.to_json()
        assert data["current_best"] is not None
        assert data["current_best"]["score"] == 5.0

    def test_solution_without_eval_serialized(self):
        state = EvoEngineerRunStateDict(task_info={})
        state.population = [Solution("unevaluated")]
        data = state.to_json()
        assert data["population"][0]["evaluation_res"] is None

    def test_invalid_solution_excluded_from_best(self):
        state = EvoEngineerRunStateDict(task_info={})
        state.sol_history = [make_sol("code", float("-inf"), valid=False)]
        data = state.to_json()
        assert data["current_best"] is None
