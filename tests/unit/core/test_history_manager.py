# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for HistoryManager."""

import os

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.core.history_manager import HistoryManager


@pytest.fixture
def hm(tmp_path):
    return HistoryManager(str(tmp_path / "history_test"))


def make_solution(code, score, valid=True):
    return Solution(
        sol_string=code,
        other_info={"tag": "test"},
        evaluation_res=EvaluationResult(valid=valid, score=score, additional_info={}),
    )


class TestHistoryManagerInit:
    def test_creates_directories(self, tmp_path):
        output = str(tmp_path / "run")
        HistoryManager(output)
        assert os.path.isdir(os.path.join(output, "history"))
        assert os.path.isdir(os.path.join(output, "summary"))

    def test_stores_output_path(self, tmp_path):
        output = str(tmp_path / "run")
        hm = HistoryManager(output)
        assert hm.output_path == output


class TestGenerationHistory:
    def test_save_and_load_roundtrip(self, hm):
        solutions = [make_solution("def f(): return 1", 1.0), make_solution("def f(): return 2", 2.0)]
        usage = [{"tokens": 100}, {"tokens": 200}]
        hm.save_generation_history(0, solutions, usage, statistics={"valid_rate": 1.0})
        data = hm.load_generation_history(0)
        assert data is not None
        assert data["generation"] == 0
        assert len(data["solutions"]) == 2
        assert data["solutions"][0]["sol_string"] == "def f(): return 1"
        assert data["solutions"][1]["evaluation_res"]["score"] == 2.0
        assert data["usage"] == usage
        assert data["statistics"]["valid_rate"] == 1.0

    def test_load_nonexistent_generation(self, hm):
        assert hm.load_generation_history(999) is None

    def test_solution_without_evaluation(self, hm):
        sol = Solution("unevaluated_code")
        hm.save_generation_history(0, [sol], [])
        data = hm.load_generation_history(0)
        assert data["solutions"][0]["evaluation_res"] is None

    def test_get_all_generations_empty(self, hm):
        assert hm.get_all_generations() == []

    def test_get_all_generations_sorted(self, hm):
        for gen in [3, 1, 0, 2]:
            hm.save_generation_history(gen, [], [])
        assert hm.get_all_generations() == [0, 1, 2, 3]

    def test_invalid_solution_saved(self, hm):
        sol = make_solution("bad_code", float("-inf"), valid=False)
        hm.save_generation_history(0, [sol], [])
        data = hm.load_generation_history(0)
        assert data["solutions"][0]["evaluation_res"]["valid"] is False


class TestBatchHistory:
    def test_save_and_load_roundtrip(self, hm):
        solutions = [make_solution("def f(): return 0", 0.0)]
        hm.save_batch_history(0, (0, 10), solutions, [{"tokens": 50}], metadata={"island": 1})
        data = hm.load_batch_history(0)
        assert data is not None
        assert data["batch_id"] == 0
        assert data["sample_range"] == [0, 10]
        assert data["metadata"]["island"] == 1
        assert len(data["solutions"]) == 1

    def test_load_nonexistent_batch(self, hm):
        assert hm.load_batch_history(999) is None

    def test_get_all_batches_empty(self, hm):
        assert hm.get_all_batches() == []

    def test_get_all_batches_sorted(self, hm):
        for batch_id in [5, 0, 2, 1]:
            hm.save_batch_history(batch_id, (0, 1), [], [])
        assert hm.get_all_batches() == [0, 1, 2, 5]


class TestSummaryHistory:
    def test_usage_history_roundtrip(self, hm):
        usage = {"gen_0": {"tokens": 1000}, "gen_1": {"tokens": 2000}}
        hm.save_usage_history(usage)
        loaded = hm.load_usage_history()
        assert loaded == usage

    def test_load_usage_history_empty(self, hm):
        assert hm.load_usage_history() == {}

    def test_best_per_generation_roundtrip(self, hm):
        best = [{"gen": 0, "score": 1.0}, {"gen": 1, "score": 2.5}]
        hm.save_best_per_generation(best)
        loaded = hm.load_best_per_generation()
        assert loaded == best

    def test_load_best_per_generation_empty(self, hm):
        assert hm.load_best_per_generation() == []
