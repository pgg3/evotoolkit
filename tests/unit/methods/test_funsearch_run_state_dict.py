# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for FunSearchRunStateDict serialization."""

import os

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.evo_method.funsearch.run_state_dict import FunSearchRunStateDict


def make_sol(code, score=1.0, valid=True):
    return Solution(
        sol_string=code,
        evaluation_res=EvaluationResult(valid=valid, score=score, additional_info={}),
    )


class TestFunSearchRunStateDictInit:
    def test_default_values(self):
        state = FunSearchRunStateDict(task_info={"name": "t"})
        assert state.tot_sample_nums == 0
        assert state.is_done is False
        assert state.sol_history == []
        assert state.database_file is None
        assert state.batch_size == 1

    def test_custom_values(self):
        state = FunSearchRunStateDict(
            task_info={},
            tot_sample_nums=10,
            is_done=True,
            batch_size=5,
        )
        assert state.tot_sample_nums == 10
        assert state.is_done is True
        assert state.batch_size == 5


class TestFunSearchRunStateDictSerialization:
    def test_to_json_contains_keys(self):
        state = FunSearchRunStateDict(task_info={})
        data = state.to_json()
        for key in ("task_info", "tot_sample_nums", "batch_size", "is_done", "current_best"):
            assert key in data

    def test_roundtrip_empty(self):
        state = FunSearchRunStateDict(task_info={"name": "t"})
        data = state.to_json()
        restored = FunSearchRunStateDict.from_json(data)
        assert restored.tot_sample_nums == 0
        assert restored.is_done is False
        assert restored.database_file is None

    def test_roundtrip_with_data(self):
        state = FunSearchRunStateDict(
            task_info={"name": "t"},
            tot_sample_nums=15,
            database_file="db.json",
            batch_size=3,
        )
        data = state.to_json()
        restored = FunSearchRunStateDict.from_json(data)
        assert restored.tot_sample_nums == 15
        assert restored.database_file == "db.json"
        assert restored.batch_size == 3

    def test_roundtrip_to_file(self, tmp_path):
        state = FunSearchRunStateDict(task_info={"name": "t"}, tot_sample_nums=5)
        file_path = str(tmp_path / "funsearch_state.json")
        state.to_json_file(file_path)
        assert os.path.exists(file_path)
        restored = FunSearchRunStateDict.from_json_file(file_path)
        assert restored.tot_sample_nums == 5

    def test_current_best_none_when_empty(self):
        state = FunSearchRunStateDict(task_info={})
        data = state.to_json()
        assert data["current_best"] is None

    def test_current_best_from_sol_history(self):
        state = FunSearchRunStateDict(task_info={})
        state.sol_history = [make_sol("code_a", 3.0), make_sol("code_b", 1.0)]
        data = state.to_json()
        assert data["current_best"]["score"] == 3.0
        assert data["current_best"]["sol_string"] == "code_a"

    def test_invalid_solutions_excluded_from_best(self):
        state = FunSearchRunStateDict(task_info={})
        state.sol_history = [make_sol("code", float("-inf"), valid=False)]
        data = state.to_json()
        assert data["current_best"] is None

    def test_save_and_load_database_state(self, tmp_path):
        state = FunSearchRunStateDict(task_info={})
        output = str(tmp_path)
        db_data = {"island_0": ["func_a", "func_b"]}
        state.save_database_state(db_data, output)
        assert state.database_file is not None
        loaded = state.load_database_state(output)
        assert loaded == db_data

    def test_load_database_state_missing_file(self, tmp_path):
        state = FunSearchRunStateDict(task_info={}, database_file="nonexistent.json")
        result = state.load_database_state(str(tmp_path))
        assert result == {}

    def test_load_database_state_no_file_set(self):
        state = FunSearchRunStateDict(task_info={})
        result = state.load_database_state("/some/path")
        assert result == {}

    def test_has_database_state_false_no_file(self):
        state = FunSearchRunStateDict(task_info={})
        assert state.has_database_state("/some/path") is False

    def test_has_database_state_true(self, tmp_path):
        state = FunSearchRunStateDict(task_info={})
        state.save_database_state({"x": 1}, str(tmp_path))
        assert state.has_database_state(str(tmp_path)) is True
