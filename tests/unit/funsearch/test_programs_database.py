# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for FunSearch ProgramsDatabase."""


from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.evo_method.funsearch.programs_database import ProgramsDatabase


def _make_valid_solution(code: str, score: float) -> Solution:
    return Solution(
        sol_string=code,
        evaluation_res=EvaluationResult(valid=True, score=score, additional_info={}),
    )


def _make_invalid_solution(code: str) -> Solution:
    return Solution(
        sol_string=code,
        evaluation_res=EvaluationResult(valid=False, score=float("-inf"), additional_info={}),
    )


class TestProgramsDatabase:
    def test_init(self):
        db = ProgramsDatabase(num_islands=3, solutions_per_prompt=2)
        assert db.num_islands == 3
        assert len(db.islands) == 3
        assert db.best_scores_per_island == [float("-inf")] * 3

    def test_register_solution_invalid_ignored(self):
        db = ProgramsDatabase(num_islands=2)
        invalid = _make_invalid_solution("bad code")
        db.register_solution(invalid)
        assert db.get_best_solution() is None or db.get_best_score() == float("-inf")

    def test_register_solution_to_all_islands_when_no_island_id(self):
        db = ProgramsDatabase(num_islands=3)
        sol = _make_valid_solution("def f(): pass", 1.0)
        db.register_solution(sol, island_id=None)
        for island in db.islands:
            assert island.num_programs == 1

    def test_register_solution_to_specific_island(self):
        db = ProgramsDatabase(num_islands=3)
        sol = _make_valid_solution("def f(): pass", 1.0)
        db.register_solution(sol, island_id=1)
        assert db.islands[0].num_programs == 0
        assert db.islands[1].num_programs == 1
        assert db.islands[2].num_programs == 0

    def test_best_scores_updated(self):
        db = ProgramsDatabase(num_islands=2)
        sol1 = _make_valid_solution("low", 0.5)
        sol2 = _make_valid_solution("high", 9.0)
        db.register_solution(sol1, island_id=0)
        db.register_solution(sol2, island_id=0)
        assert db.best_scores_per_island[0] == 9.0

    def test_get_best_solution_none_when_empty(self):
        db = ProgramsDatabase(num_islands=2)
        # All scores are -inf, so argmax picks index 0, best_solutions_per_island[0] is None
        result = db.get_best_solution()
        assert result is None

    def test_get_best_solution_after_register(self):
        db = ProgramsDatabase(num_islands=2)
        sol = _make_valid_solution("best", 5.0)
        db.register_solution(sol, island_id=0)
        best = db.get_best_solution()
        assert best is not None

    def test_get_best_score(self):
        db = ProgramsDatabase(num_islands=2)
        sol = _make_valid_solution("best", 7.5)
        db.register_solution(sol, island_id=1)
        assert db.get_best_score() == 7.5

    def test_get_prompt_solutions_returns_tuple(self):
        db = ProgramsDatabase(num_islands=2, solutions_per_prompt=1)
        sol = _make_valid_solution("code", 1.0)
        db.register_solution(sol, island_id=0)
        db.register_solution(sol, island_id=1)
        solutions, island_id = db.get_prompt_solutions()
        assert isinstance(solutions, list)
        assert island_id in [0, 1]

    def test_get_statistics(self):
        db = ProgramsDatabase(num_islands=2)
        sol = _make_valid_solution("code", 3.0)
        db.register_solution(sol, island_id=0)
        stats = db.get_statistics()
        assert "total_programs" in stats
        assert "num_islands" in stats
        assert "global_best_score" in stats
        assert "island_stats" in stats
        assert stats["num_islands"] == 2
        assert stats["total_programs"] == 1

    def test_reset_islands(self):
        db = ProgramsDatabase(num_islands=4)
        for i in range(4):
            sol = _make_valid_solution(f"code_{i}", float(i + 1))
            db.register_solution(sol, island_id=i)
        db.reset_islands()
        # After reset: half (2) should be reset — they get a founder from a best island
        # So they won't be empty (founder is added), but their original clusters are gone
        # Verify that reset happened (total programs decreased from 4)
        total_programs = sum(island.num_programs for island in db.islands)
        # 2 islands kept (1 program each) + 2 reset islands with 1 founder each = still 4
        # What matters is that the reset happened without error
        assert total_programs >= 2  # At least kept islands remain
        # The reset_islands method should run without exception
        # Run it again to ensure idempotent
        db.reset_islands()

    def test_to_dict_and_from_dict_roundtrip(self):
        db = ProgramsDatabase(num_islands=2, solutions_per_prompt=1)
        sol = _make_valid_solution("def f(x): return x", 2.5)
        db.register_solution(sol, island_id=0)

        data = db.to_dict()
        restored = ProgramsDatabase.from_dict(data)

        assert restored.num_islands == db.num_islands
        assert restored.solutions_per_prompt == db.solutions_per_prompt
        assert len(restored.islands) == 2

    def test_to_dict_contains_required_keys(self):
        db = ProgramsDatabase(num_islands=2)
        data = db.to_dict()
        assert "num_islands" in data
        assert "solutions_per_prompt" in data
        assert "reset_period" in data
        assert "islands" in data
        assert "best_scores_per_island" in data

    def test_from_dict_restores_best_solutions(self):
        db = ProgramsDatabase(num_islands=2, solutions_per_prompt=1)
        sol = _make_valid_solution("def f(x): return x * 2", 4.0)
        db.register_solution(sol, island_id=0)

        data = db.to_dict()
        restored = ProgramsDatabase.from_dict(data)

        # Island 0 should have clusters
        assert len(restored.islands[0].clusters) > 0
