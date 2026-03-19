# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for HistoryManager — covering uncovered lines (75, 82-83, 140, 147-148)."""

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.core.history_manager import HistoryManager


def _make_solution(code: str, score: float, valid: bool = True) -> Solution:
    return Solution(
        sol_string=code,
        evaluation_res=EvaluationResult(valid=valid, score=score, additional_info={}),
    )


class TestHistoryManagerGenerations:
    def test_save_and_load_generation(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        sol = _make_solution("def f(x): return x", 1.0)
        hm.save_generation_history(generation=0, solutions=[sol], usage=[{"tokens": 100}])
        loaded = hm.load_generation_history(0)
        assert loaded is not None
        assert loaded["generation"] == 0
        assert len(loaded["solutions"]) == 1

    def test_load_nonexistent_generation_returns_none(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        result = hm.load_generation_history(999)
        assert result is None

    def test_get_all_generations_empty(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        gens = hm.get_all_generations()
        assert gens == []

    def test_get_all_generations_after_save(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        sol = _make_solution("code", 1.0)
        hm.save_generation_history(generation=2, solutions=[sol], usage=[])
        hm.save_generation_history(generation=5, solutions=[sol], usage=[])
        gens = hm.get_all_generations()
        assert 2 in gens
        assert 5 in gens
        assert gens == sorted(gens)

    def test_get_all_generations_ignores_non_gen_files(self, tmp_path):
        """get_all_generations should skip files that don't match gen_N.json."""
        hm = HistoryManager(str(tmp_path))
        # Create a non-gen file in history dir
        other_file = tmp_path / "history" / "batch_0001.json"
        other_file.write_text("{}")
        gens = hm.get_all_generations()
        assert gens == []

    def test_get_all_generations_nonexistent_dir(self, tmp_path):
        """When history_dir doesn't exist, returns empty list."""
        import shutil

        hm = HistoryManager(str(tmp_path))
        shutil.rmtree(hm.history_dir)
        gens = hm.get_all_generations()
        assert gens == []


class TestHistoryManagerBatches:
    def test_save_and_load_batch(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        sol = _make_solution("code", 2.0)
        hm.save_batch_history(
            batch_id=0,
            sample_range=(0, 5),
            solutions=[sol],
            usage=[{"tokens": 50}],
        )
        loaded = hm.load_batch_history(0)
        assert loaded is not None
        assert loaded["batch_id"] == 0

    def test_load_nonexistent_batch_returns_none(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        result = hm.load_batch_history(999)
        assert result is None

    def test_get_all_batches_empty(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        batches = hm.get_all_batches()
        assert batches == []

    def test_get_all_batches_after_save(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        sol = _make_solution("code", 1.0)
        hm.save_batch_history(batch_id=0, sample_range=(0, 1), solutions=[sol], usage=[])
        hm.save_batch_history(batch_id=1, sample_range=(1, 2), solutions=[sol], usage=[])
        batches = hm.get_all_batches()
        assert 0 in batches
        assert 1 in batches

    def test_get_all_batches_ignores_non_batch_files(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        other_file = tmp_path / "history" / "gen_0.json"
        other_file.write_text("{}")
        batches = hm.get_all_batches()
        assert batches == []

    def test_get_all_batches_nonexistent_dir(self, tmp_path):
        import shutil

        hm = HistoryManager(str(tmp_path))
        shutil.rmtree(hm.history_dir)
        batches = hm.get_all_batches()
        assert batches == []


class TestHistoryManagerSummary:
    def test_save_and_load_usage_history(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        usage = {"sample": [{"tokens": 100}, {"tokens": 200}]}
        hm.save_usage_history(usage)
        loaded = hm.load_usage_history()
        assert loaded["sample"][0]["tokens"] == 100

    def test_load_usage_history_no_file(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        result = hm.load_usage_history()
        assert result == {}

    def test_save_and_load_best_per_generation(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        best_list = [{"generation": 0, "score": 1.0}, {"generation": 1, "score": 2.0}]
        hm.save_best_per_generation(best_list)
        loaded = hm.load_best_per_generation()
        assert len(loaded) == 2
        assert loaded[0]["score"] == 1.0

    def test_load_best_per_generation_no_file(self, tmp_path):
        hm = HistoryManager(str(tmp_path))
        result = hm.load_best_per_generation()
        assert result == []

    def test_save_solution_without_evaluation_res(self, tmp_path):
        """Solutions without evaluation_res should be saved with evaluation_res=None."""
        hm = HistoryManager(str(tmp_path))
        sol = Solution("code_no_eval")
        hm.save_generation_history(generation=0, solutions=[sol], usage=[])
        loaded = hm.load_generation_history(0)
        assert loaded["solutions"][0]["evaluation_res"] is None
