# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EvoEngineer and FunSearch run state dicts (missing coverage)."""


from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.evo_method.evoengineer.run_state_dict import EvoEngineerRunStateDict
from evotoolkit.evo_method.funsearch.run_state_dict import FunSearchRunStateDict


def _make_valid_solution(code: str = "def f(x): return x", score: float = 1.0) -> Solution:
    return Solution(
        sol_string=code,
        evaluation_res=EvaluationResult(valid=True, score=score, additional_info={}),
    )


def _make_invalid_solution() -> Solution:
    return Solution(
        sol_string="bad",
        evaluation_res=EvaluationResult(valid=False, score=float("-inf"), additional_info={}),
    )


# ---------------------------------------------------------------------------
# EvoEngineerRunStateDict
# ---------------------------------------------------------------------------


class TestEvoEngineerRunStateDict:
    def test_init_defaults(self):
        rsd = EvoEngineerRunStateDict(task_info={"name": "test"})
        assert rsd.generation == 0
        assert rsd.tot_sample_nums == 0
        assert rsd.is_done is False
        assert rsd.sol_history == []
        assert rsd.population == []
        assert rsd.usage_history == {}
        assert rsd.current_gen_solutions == []
        assert rsd.current_gen_usage == []

    def test_to_json_empty(self):
        rsd = EvoEngineerRunStateDict(task_info={"name": "test"})
        data = rsd.to_json()
        assert "generation" in data
        assert "population" in data
        assert "is_done" in data
        assert data["current_best"] is None

    def test_to_json_with_valid_solution(self):
        rsd = EvoEngineerRunStateDict(task_info={})
        sol = _make_valid_solution(score=3.5)
        rsd.sol_history.append(sol)
        data = rsd.to_json()
        assert data["current_best"] is not None
        assert data["current_best"]["score"] == 3.5

    def test_to_json_with_only_invalid_solutions(self):
        rsd = EvoEngineerRunStateDict(task_info={})
        rsd.sol_history.append(_make_invalid_solution())
        data = rsd.to_json()
        assert data["current_best"] is None

    def test_to_json_population_serialization(self):
        rsd = EvoEngineerRunStateDict(task_info={})
        sol = _make_valid_solution("def f(x): return x * 2", 2.0)
        rsd.population.append(sol)
        data = rsd.to_json()
        assert len(data["population"]) == 1
        pop_entry = data["population"][0]
        assert pop_entry["sol_string"] == "def f(x): return x * 2"
        assert pop_entry["evaluation_res"]["score"] == 2.0

    def test_from_json_roundtrip(self):
        rsd = EvoEngineerRunStateDict(task_info={"key": "val"}, generation=3, tot_sample_nums=10)
        sol = _make_valid_solution()
        rsd.population.append(sol)
        data = rsd.to_json()
        restored = EvoEngineerRunStateDict.from_json(data)
        assert restored.generation == 3
        assert restored.tot_sample_nums == 10
        assert len(restored.population) == 1
        assert restored.population[0].sol_string == sol.sol_string

    def test_from_json_solution_with_evaluation_res(self):
        rsd = EvoEngineerRunStateDict(task_info={})
        sol = _make_valid_solution(score=5.0)
        rsd.population.append(sol)
        data = rsd.to_json()
        restored = EvoEngineerRunStateDict.from_json(data)
        assert restored.population[0].evaluation_res is not None
        assert restored.population[0].evaluation_res.score == 5.0

    def test_save_current_history_no_manager(self):
        """save_current_history is no-op without history manager."""
        rsd = EvoEngineerRunStateDict(task_info={})
        sol = _make_valid_solution()
        rsd.current_gen_solutions.append(sol)
        rsd.save_current_history()  # Should not raise

    def test_save_current_history_empty_solutions(self, tmp_path):
        rsd = EvoEngineerRunStateDict(task_info={})
        rsd.init_history_manager(str(tmp_path))
        rsd.save_current_history()  # Empty, should not raise or write

    def test_save_current_history_with_solutions(self, tmp_path):
        rsd = EvoEngineerRunStateDict(task_info={}, generation=2)
        rsd.init_history_manager(str(tmp_path))
        sol = _make_valid_solution(score=1.5)
        rsd.current_gen_solutions.append(sol)
        rsd.current_gen_usage.append({"tokens": 100})
        rsd.usage_history["sample"] = [{"tokens": 100}]
        rsd.save_current_history()
        # After save, current gen should be cleared
        assert rsd.current_gen_solutions == []
        assert rsd.current_gen_usage == []

    def test_save_current_history_statistics_with_valid_sols(self, tmp_path):
        rsd = EvoEngineerRunStateDict(task_info={}, generation=1)
        rsd.init_history_manager(str(tmp_path))
        sol1 = _make_valid_solution(score=2.0)
        sol2 = _make_valid_solution(score=4.0)
        rsd.current_gen_solutions = [sol1, sol2]
        rsd.usage_history["sample"] = []
        rsd.save_current_history()
        # History file should exist
        history_dir = tmp_path / "history"
        files = list(history_dir.iterdir())
        assert len(files) == 1


# ---------------------------------------------------------------------------
# FunSearchRunStateDict
# ---------------------------------------------------------------------------


class TestFunSearchRunStateDict:
    def test_init_defaults(self):
        rsd = FunSearchRunStateDict(task_info={"name": "test"})
        assert rsd.tot_sample_nums == 0
        assert rsd.sol_history == []
        assert rsd.database_file is None
        assert rsd.is_done is False

    def test_to_json_empty(self):
        rsd = FunSearchRunStateDict(task_info={})
        data = rsd.to_json()
        assert "tot_sample_nums" in data
        assert "is_done" in data
        assert data["current_best"] is None

    def test_to_json_with_valid_solution(self):
        rsd = FunSearchRunStateDict(task_info={})
        sol = _make_valid_solution(score=7.0)
        rsd.sol_history.append(sol)
        data = rsd.to_json()
        assert data["current_best"]["score"] == 7.0

    def test_from_json_roundtrip(self):
        rsd = FunSearchRunStateDict(task_info={"x": 1}, tot_sample_nums=5, is_done=True)
        data = rsd.to_json()
        restored = FunSearchRunStateDict.from_json(data)
        assert restored.tot_sample_nums == 5
        assert restored.is_done is True

    def test_save_database_state(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={})
        output_path = str(tmp_path)
        db_dict = {"num_islands": 2, "islands": []}
        rsd.save_database_state(db_dict, output_path)
        assert rsd.database_file is not None
        import os

        assert os.path.exists(rsd.database_file)

    def test_load_database_state_no_file(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={})
        result = rsd.load_database_state(str(tmp_path))
        assert result == {}

    def test_load_database_state_after_save(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={})
        db_dict = {"num_islands": 3, "solutions_per_prompt": 2, "islands": []}
        rsd.save_database_state(db_dict, str(tmp_path))
        loaded = rsd.load_database_state(str(tmp_path))
        assert loaded["num_islands"] == 3

    def test_has_database_state_false_when_no_file(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={})
        assert rsd.has_database_state(str(tmp_path)) is False

    def test_has_database_state_true_after_save(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={})
        rsd.save_database_state({}, str(tmp_path))
        assert rsd.has_database_state(str(tmp_path)) is True

    def test_save_current_history_no_manager(self):
        rsd = FunSearchRunStateDict(task_info={})
        sol = _make_valid_solution()
        rsd.current_batch_solutions.append(sol)
        rsd.save_current_history()  # No-op without manager

    def test_save_current_history_empty_batch(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={})
        rsd.init_history_manager(str(tmp_path))
        rsd.save_current_history()  # Empty, no-op

    def test_save_current_history_below_batch_size(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={}, batch_size=5)
        rsd.init_history_manager(str(tmp_path))
        sol = _make_valid_solution()
        rsd.current_batch_solutions.append(sol)
        rsd.save_current_history()  # 1 < 5, not done, should not save
        history_dir = tmp_path / "history"
        files = list(history_dir.iterdir())
        assert len(files) == 0

    def test_save_current_history_when_done(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={}, batch_size=5, tot_sample_nums=1)
        rsd.init_history_manager(str(tmp_path))
        rsd.is_done = True
        sol = _make_valid_solution()
        rsd.current_batch_solutions.append(sol)
        rsd.usage_history["sample"] = []
        rsd.save_current_history()
        # Should have saved since is_done=True
        history_dir = tmp_path / "history"
        files = list(history_dir.iterdir())
        assert len(files) == 1

    def test_save_current_history_clears_batch(self, tmp_path):
        rsd = FunSearchRunStateDict(task_info={}, batch_size=1, tot_sample_nums=1)
        rsd.init_history_manager(str(tmp_path))
        sol = _make_valid_solution()
        rsd.current_batch_solutions.append(sol)
        rsd.usage_history["sample"] = []
        rsd.save_current_history()
        assert rsd.current_batch_solutions == []
        assert rsd.current_batch_usage == []

    def test_load_database_state_corrupt_file(self, tmp_path):
        """Corrupt database file should return empty dict."""

        rsd = FunSearchRunStateDict(task_info={})
        rsd.database_file = str(tmp_path / "db.json")
        with open(rsd.database_file, "w") as f:
            f.write("NOT VALID JSON{{{")
        result = rsd.load_database_state(str(tmp_path))
        assert result == {}
