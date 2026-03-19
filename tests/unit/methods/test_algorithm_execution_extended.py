# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Additional branch coverage tests for built-in algorithm execution."""

from __future__ import annotations

import builtins

import pytest

from evotoolkit.core import EvaluationResult, Operator, Solution
from evotoolkit.evo_method.eoh import EoH, EoHConfig
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig
from evotoolkit.evo_method.funsearch import FunSearch, FunSearchConfig
from evotoolkit.task.string_optimization.method_interface.funsearch_interface import (
    FunSearchStringInterface,
)
from evotoolkit.task.string_optimization.string_task import StringTask


def _make_solution(
    score: float | None = 1.0,
    *,
    valid: bool = True,
    sol_string: str = "def f(x):\n    return x + 1\n",
    name: str = "candidate",
    thought: str = "simple idea",
) -> Solution:
    evaluation = None
    if score is not None or valid:
        evaluation = EvaluationResult(
            valid=valid,
            score=score if score is not None else float("-inf"),
            additional_info={"success_rate": 0.5},
        )
    return Solution(sol_string=sol_string, evaluation_res=evaluation, other_info={"name": name, "thought": thought})


class StaticLLM:
    def __init__(self, response: str = "mock"):
        self.response = response

    def get_response(self, messages, **kwargs):
        return self.response, {"prompt_tokens": 1}


class BrokenLLM:
    def get_response(self, messages, **kwargs):
        raise RuntimeError("llm failure")


class MinimalStringTask(StringTask):
    def _process_data(self, data):
        self.data = data
        self.task_info = {"name": "string"}

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        return EvaluationResult(valid=bool(candidate_string), score=float(len(candidate_string)), additional_info={})

    def get_base_task_description(self) -> str:
        return "Optimize a string."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("baseline")


class ImmediateFuture:
    def __init__(self, fn, *args, **kwargs):
        self._exception = None
        self._result = None
        try:
            self._result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exercised via result()
            self._exception = exc

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


class ImmediateExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return ImmediateFuture(fn, *args, **kwargs)


def _install_immediate_executor(monkeypatch, module_path: str):
    monkeypatch.setattr(f"{module_path}.concurrent.futures.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(f"{module_path}.concurrent.futures.as_completed", lambda futures: futures)


class TestEoHExtendedExecution:
    def test_run_exits_when_initial_solution_cannot_be_created(self, eoh_interface, tmp_output, monkeypatch):
        cfg = EoHConfig(interface=eoh_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False)
        algo = EoH(cfg)

        monkeypatch.setattr(algo, "_get_init_sol", lambda: None)
        monkeypatch.setattr(builtins, "exit", lambda: (_ for _ in ()).throw(SystemExit("eoh-init-failed")))

        with pytest.raises(SystemExit, match="eoh-init-failed"):
            algo.run()

    def test_run_returns_when_initialization_never_reaches_selection_size(self, eoh_interface, tmp_output, monkeypatch):
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=StaticLLM(),
            verbose=False,
            selection_num=2,
            pop_size=2,
            max_generations=1,
            max_sample_nums=1,
        )
        algo = EoH(cfg)

        monkeypatch.setattr(algo, "_initialize_population", lambda: None)

        result = algo.run()

        assert result is None
        assert algo.run_state_dict.is_done is False

    def test_run_handles_operator_runtime_error_and_still_finishes(self, eoh_interface, tmp_output, monkeypatch):
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=StaticLLM(),
            verbose=False,
            selection_num=1,
            pop_size=2,
            max_generations=2,
            max_sample_nums=4,
        )
        algo = EoH(cfg)
        algo.run_state_dict.sol_history = [_make_solution()]
        algo.run_state_dict.population = [_make_solution()]
        algo.run_state_dict.generation = 1

        def fail_once():
            algo.run_state_dict.tot_sample_nums = cfg.max_sample_nums
            raise RuntimeError("operator failure")

        monkeypatch.setattr(algo, "_apply_operators_parallel", fail_once)

        algo.run()

        assert algo.run_state_dict.is_done is True

    def test_run_handles_keyboard_interrupt_and_marks_done(self, eoh_interface, tmp_output, monkeypatch):
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=StaticLLM(),
            verbose=False,
            selection_num=1,
            pop_size=2,
            max_generations=2,
            max_sample_nums=4,
        )
        algo = EoH(cfg)
        algo.run_state_dict.sol_history = [_make_solution()]
        algo.run_state_dict.population = [_make_solution()]
        algo.run_state_dict.generation = 1
        monkeypatch.setattr(algo, "_apply_operators_parallel", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

        algo.run()

        assert algo.run_state_dict.is_done is True

    def test_initialize_population_warns_when_only_invalid_solutions_are_generated(self, eoh_interface, tmp_output, monkeypatch):
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=StaticLLM(),
            verbose=False,
            selection_num=1,
            pop_size=2,
            max_sample_nums=1,
        )
        algo = EoH(cfg)
        invalid = _make_solution(valid=False, sol_string="def bad():\n    pass\n")
        monkeypatch.setattr(algo, "_generate_and_evaluate_initial_solutions", lambda: [invalid])

        algo._initialize_population()

        assert algo.run_state_dict.generation == 0
        assert algo.run_state_dict.tot_sample_nums == 1

    def test_generate_and_evaluate_initial_solutions_handles_generation_failures(self, eoh_interface, tmp_output, monkeypatch):
        _install_immediate_executor(monkeypatch, "evotoolkit.evo_method.eoh.eoh")
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=StaticLLM(),
            verbose=False,
            num_samplers=2,
            num_evaluators=1,
        )
        algo = EoH(cfg)
        algo.run_state_dict.usage_history["sample"] = []

        def fake_generate(sampler_id: int):
            if sampler_id == 0:
                raise RuntimeError("generation failed")
            return Solution("def f(x):\n    return x\n"), {"sampler": sampler_id}

        monkeypatch.setattr(algo, "_generate_single_initial_solution", fake_generate)

        evaluated = algo._generate_and_evaluate_initial_solutions()

        assert len(evaluated) == 1
        assert evaluated[0].evaluation_res is not None

    def test_generate_and_evaluate_initial_solutions_keeps_solution_when_eval_fails(self, eoh_interface, tmp_output, monkeypatch):
        _install_immediate_executor(monkeypatch, "evotoolkit.evo_method.eoh.eoh")
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=StaticLLM(),
            verbose=False,
            num_samplers=1,
            num_evaluators=1,
        )
        algo = EoH(cfg)
        algo.run_state_dict.usage_history["sample"] = []
        monkeypatch.setattr(
            algo,
            "_generate_single_initial_solution",
            lambda sampler_id: (Solution("def f(x):\n    return x\n"), {"sampler": sampler_id}),
        )

        def fail_eval(code: str):
            raise RuntimeError("eval failed")

        monkeypatch.setattr(cfg.task, "evaluate_code", fail_eval)

        evaluated = algo._generate_and_evaluate_initial_solutions()

        assert len(evaluated) == 1
        assert evaluated[0].evaluation_res is None

    def test_evaluate_solutions_handles_empty_candidates_and_eval_errors(self, eoh_interface, tmp_output, monkeypatch):
        cfg = EoHConfig(interface=eoh_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, num_evaluators=2)
        algo = EoH(cfg)
        solutions = [
            Solution("def f(x):\n    return x\n"),
            Solution(""),
            Solution("def f(x):\n    return x + 1\n"),
        ]

        def fake_evaluate(code: str):
            if "+ 1" in code:
                raise RuntimeError("bad branch")
            return EvaluationResult(valid=True, score=3.0, additional_info={})

        monkeypatch.setattr(cfg.task, "evaluate_code", fake_evaluate)

        result = algo._evaluate_solutions(solutions)

        assert result[0].evaluation_res.score == 3.0
        assert result[1].evaluation_res is None
        assert result[2].evaluation_res is None

    def test_get_best_valid_sol_falls_back_to_last_or_none(self, eoh_interface, tmp_output):
        cfg = EoHConfig(interface=eoh_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False)
        algo = EoH(cfg)
        invalids = [_make_solution(valid=False, sol_string="bad1"), _make_solution(valid=False, sol_string="bad2")]

        assert algo._get_best_valid_sol(invalids).sol_string == "bad2"
        assert algo._get_best_valid_sol([]) is None

    def test_apply_operators_parallel_handles_optional_operators_and_failures(self, eoh_interface, tmp_output, monkeypatch):
        _install_immediate_executor(monkeypatch, "evotoolkit.evo_method.eoh.eoh")
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=StaticLLM(),
            verbose=False,
            selection_num=1,
            use_e2_operator=True,
            use_m1_operator=True,
            use_m2_operator=True,
            num_samplers=4,
            num_evaluators=1,
        )
        algo = EoH(cfg)
        base = _make_solution(score=10.0)
        algo.run_state_dict.population = [base]
        algo.run_state_dict.usage_history["sample"] = []
        monkeypatch.setattr(algo, "_select_individuals", lambda num: [base] if num > 0 else [])

        def fake_generate(prompt_content, operator_type, sampler_id):
            if operator_type == "E1":
                return Solution(""), {"operator": operator_type}
            if operator_type == "E2":
                raise RuntimeError("generation boom")
            if operator_type == "M1":
                return Solution("raise_eval_error"), {"operator": operator_type}
            return Solution("valid_solution"), {"operator": operator_type}

        def fake_evaluate(code: str):
            if code == "raise_eval_error":
                raise RuntimeError("eval boom")
            return EvaluationResult(valid=True, score=5.0, additional_info={})

        monkeypatch.setattr(algo, "_generate_single_operator_solution", fake_generate)
        monkeypatch.setattr(cfg.task, "evaluate_code", fake_evaluate)

        generated = algo._apply_operators_parallel()

        assert len(generated) == 3
        assert any(sol.sol_string == "" for sol in generated)
        assert any(sol.sol_string == "raise_eval_error" and sol.evaluation_res is None for sol in generated)
        assert any(sol.sol_string == "valid_solution" and sol.evaluation_res.score == 5.0 for sol in generated)

    def test_manage_population_size_keeps_recent_invalid_when_slots_remain(self, eoh_interface, tmp_output):
        cfg = EoHConfig(interface=eoh_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, pop_size=3)
        algo = EoH(cfg)
        best = _make_solution(score=10.0, name="best")
        invalid_1 = _make_solution(valid=False, score=float("-inf"), sol_string="bad1")
        invalid_2 = _make_solution(valid=False, score=float("-inf"), sol_string="bad2")
        invalid_3 = _make_solution(valid=False, score=float("-inf"), sol_string="bad3")
        algo.run_state_dict.population = [best, invalid_1, invalid_2, invalid_3]

        algo._manage_population_size()

        assert len(algo.run_state_dict.population) == 3
        assert algo.run_state_dict.population[0].other_info["name"] == "best"
        assert [sol.sol_string for sol in algo.run_state_dict.population[1:]] == ["bad2", "bad3"]

    def test_manage_population_size_returns_immediately_when_population_is_small(self, eoh_interface, tmp_output):
        cfg = EoHConfig(interface=eoh_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, pop_size=5)
        algo = EoH(cfg)
        algo.run_state_dict.population = [_make_solution()]

        algo._manage_population_size()

        assert len(algo.run_state_dict.population) == 1

    def test_select_individuals_handles_zero_and_fallback_cases(self, eoh_interface, tmp_output):
        cfg = EoHConfig(interface=eoh_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False)
        algo = EoH(cfg)
        fallback_population = [
            _make_solution(valid=False, sol_string="first"),
            _make_solution(valid=False, sol_string="second"),
        ]
        algo.run_state_dict.population = fallback_population

        assert algo._select_individuals(0) == []
        assert algo._select_individuals(2) == fallback_population

    def test_generate_single_operator_solution_handles_llm_errors(self, eoh_interface, tmp_output):
        cfg = EoHConfig(interface=eoh_interface, output_path=tmp_output, running_llm=BrokenLLM(), verbose=False)
        algo = EoH(cfg)

        solution, usage = algo._generate_single_operator_solution([], "E1", 0)

        assert solution.sol_string == ""
        assert usage == {}


class TestEvoEngineerExtendedExecution:
    def test_run_exits_when_initial_solution_cannot_be_created(self, evoengineer_interface, tmp_output, monkeypatch):
        cfg = EvoEngineerConfig(interface=evoengineer_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False)
        algo = EvoEngineer(cfg)
        monkeypatch.setattr(algo, "_get_init_sol", lambda: None)
        monkeypatch.setattr(builtins, "exit", lambda: (_ for _ in ()).throw(SystemExit("evo-init-failed")))

        with pytest.raises(SystemExit, match="evo-init-failed"):
            algo.run()

    def test_run_returns_when_initialization_never_reaches_valid_requirement(self, evoengineer_interface, tmp_output, monkeypatch):
        cfg = EvoEngineerConfig(interface=evoengineer_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, max_sample_nums=1)
        algo = EvoEngineer(cfg)
        monkeypatch.setattr(algo, "_initialize_population", lambda: None)

        result = algo.run()

        assert result is None
        assert algo.run_state_dict.is_done is False

    def test_run_handles_keyboard_interrupt_and_errors(self, evoengineer_interface, tmp_output, monkeypatch):
        cfg = EvoEngineerConfig(interface=evoengineer_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, max_generations=2)
        algo = EvoEngineer(cfg)
        valid_population = [_make_solution(score=1.0), _make_solution(score=2.0)]
        algo.run_state_dict.sol_history = list(valid_population)
        algo.run_state_dict.population = list(valid_population)
        algo.run_state_dict.generation = 1

        monkeypatch.setattr(algo, "_apply_operators_parallel", lambda operators, label="": (_ for _ in ()).throw(KeyboardInterrupt()))
        algo.run()
        assert algo.run_state_dict.is_done is True

        algo.run_state_dict.is_done = False
        algo.run_state_dict.generation = 1
        algo.run_state_dict.tot_sample_nums = 0

        def fail_with_runtime(operators, label=""):
            algo.run_state_dict.tot_sample_nums = cfg.max_sample_nums
            raise RuntimeError("operator failure")

        monkeypatch.setattr(algo, "_apply_operators_parallel", fail_with_runtime)
        algo.run()
        assert algo.run_state_dict.is_done is True

    def test_initialize_population_warns_when_requirement_not_met(self, evoengineer_interface, tmp_output, monkeypatch):
        cfg = EvoEngineerConfig(interface=evoengineer_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, max_sample_nums=1)
        algo = EvoEngineer(cfg)
        monkeypatch.setattr(algo, "_apply_operators_parallel", lambda operators, generation_label="": None)

        algo._initialize_population()

        assert algo.run_state_dict.generation == 0

    def test_get_best_valid_sol_returns_none_without_valid_solutions(self, evoengineer_interface, tmp_output):
        cfg = EvoEngineerConfig(interface=evoengineer_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False)
        algo = EvoEngineer(cfg)

        assert algo._get_best_valid_sol([_make_solution(valid=False)]) is None

    def test_apply_operators_parallel_returns_early_for_empty_operator_list(self, evoengineer_interface, tmp_output):
        cfg = EvoEngineerConfig(interface=evoengineer_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False)
        algo = EvoEngineer(cfg)

        assert algo._apply_operators_parallel([]) is None

    def test_apply_operators_parallel_handles_empty_generation_and_eval_error(self, evoengineer_interface, tmp_output, monkeypatch):
        _install_immediate_executor(monkeypatch, "evotoolkit.evo_method.evoengineer.evoengineer")
        cfg = EvoEngineerConfig(interface=evoengineer_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, num_samplers=2, num_evaluators=1)
        algo = EvoEngineer(cfg)
        parent = _make_solution(score=4.0)
        algo.run_state_dict.population = [parent]
        algo.run_state_dict.usage_history["sample"] = []
        operators = [Operator("mutation", 1), Operator("crossover", 2)]

        def fake_select(operator):
            return [parent] * max(1, operator.selection_size)

        def fake_generate(operator, selected, sampler_id):
            if operator.name == "mutation":
                return Solution(""), {"operator": operator.name}
            return Solution("raise_eval_error"), {"operator": operator.name}

        monkeypatch.setattr(algo, "_select_individuals_for_operator", fake_select)
        monkeypatch.setattr(algo, "_generate_single_solution", fake_generate)
        monkeypatch.setattr(cfg.task, "evaluate_code", lambda code: (_ for _ in ()).throw(RuntimeError("eval failed")))

        algo._apply_operators_parallel(operators, "Gen 1")

        assert algo.run_state_dict.tot_sample_nums == 2
        assert len(algo.run_state_dict.population) == 3

    def test_manage_population_and_selection_edge_cases(self, evoengineer_interface, tmp_output):
        cfg = EvoEngineerConfig(interface=evoengineer_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, pop_size=3)
        algo = EvoEngineer(cfg)
        best = _make_solution(score=9.0)
        invalid_1 = _make_solution(valid=False, sol_string="invalid1")
        invalid_2 = _make_solution(valid=False, sol_string="invalid2")
        invalid_3 = _make_solution(valid=False, sol_string="invalid3")
        algo.run_state_dict.population = [best, invalid_1, invalid_2, invalid_3]

        algo._manage_population_size()

        assert len(algo.run_state_dict.population) == 3
        assert algo._select_individuals_for_operator(Operator("init", 0)) == []
        algo.run_state_dict.population = [invalid_1, invalid_2]
        assert algo._select_individuals_for_operator(Operator("mutation", 2)) == [invalid_1, invalid_2]


class TestFunSearchExtendedExecution:
    def test_run_exits_when_initial_solution_cannot_be_created(self, funsearch_interface, tmp_output, monkeypatch):
        cfg = FunSearchConfig(interface=funsearch_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, max_sample_nums=0)
        algo = FunSearch(cfg)
        monkeypatch.setattr(algo, "_get_init_sol", lambda: None)
        monkeypatch.setattr(builtins, "exit", lambda: (_ for _ in ()).throw(SystemExit("funsearch-init-failed")))

        with pytest.raises(SystemExit, match="funsearch-init-failed"):
            algo.run()

    def test_run_restores_new_database_when_saved_state_is_unloadable(self, funsearch_interface, tmp_output, monkeypatch):
        cfg = FunSearchConfig(interface=funsearch_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, max_sample_nums=0, num_islands=2)
        algo = FunSearch(cfg)
        monkeypatch.setattr(algo.run_state_dict, "has_database_state", lambda output_path: True)
        monkeypatch.setattr(algo.run_state_dict, "load_database_state", lambda output_path: None)

        algo.run()

        assert algo.run_state_dict.is_done is True

    def test_run_rebuilds_database_from_solution_history_when_database_file_is_missing(self, funsearch_interface, tmp_output, monkeypatch):
        cfg = FunSearchConfig(interface=funsearch_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, max_sample_nums=0, num_islands=2)
        algo = FunSearch(cfg)
        valid = Solution("good", evaluation_res=EvaluationResult(valid=True, score=5.0, additional_info={}))
        invalid = Solution("bad", evaluation_res=EvaluationResult(valid=False, score=float("-inf"), additional_info={}))
        algo.run_state_dict.sol_history = [valid, invalid]
        calls = []
        monkeypatch.setattr(algo.run_state_dict, "has_database_state", lambda output_path: False)

        class FakeProgramsDatabase:
            def __init__(self, *args, **kwargs):
                self.solutions = []

            @classmethod
            def from_dict(cls, data):
                return cls()

            def register_solution(self, solution, island_id=None):
                calls.append((solution.sol_string, island_id))
                self.solutions.append(solution)

            def get_prompt_solutions(self):
                return [], 0

            def get_best_solution(self):
                return None

            def get_statistics(self):
                return {"total_programs": len(self.solutions), "num_islands": 2, "global_best_score": 0.0}

            def to_dict(self):
                return {"solutions": len(self.solutions)}

        monkeypatch.setattr("evotoolkit.evo_method.funsearch.funsearch.ProgramsDatabase", FakeProgramsDatabase)

        algo.run()

        assert calls == [("good", None)]

    def test_run_handles_empty_prompt_generation_failures_and_invalid_programs(self, funsearch_interface, tmp_output, monkeypatch):
        _install_immediate_executor(monkeypatch, "evotoolkit.evo_method.funsearch.funsearch")
        cfg = FunSearchConfig(interface=funsearch_interface, output_path=tmp_output, running_llm=StaticLLM(), verbose=False, max_sample_nums=2, num_islands=1, num_samplers=1, num_evaluators=1)

        class FakeProgramsDatabase:
            def __init__(self, *args, **kwargs):
                self.calls = 0
                self.registered = []

            @classmethod
            def from_dict(cls, data):
                return cls()

            def register_solution(self, solution, island_id=None):
                self.registered.append((solution.sol_string, island_id))

            def get_prompt_solutions(self):
                self.calls += 1
                if self.calls == 1:
                    return [], 0
                return [Solution("prompt", evaluation_res=EvaluationResult(valid=True, score=1.0, additional_info={}))], 0

            def get_best_solution(self):
                return None

            def get_statistics(self):
                return {"total_programs": len(self.registered), "num_islands": 1, "global_best_score": 0.0}

            def to_dict(self):
                return {"registered": len(self.registered)}

        monkeypatch.setattr("evotoolkit.evo_method.funsearch.funsearch.ProgramsDatabase", FakeProgramsDatabase)
        algo = FunSearch(cfg)
        algo.run_state_dict.sol_history = [_make_solution(score=3.0)]
        state = {"calls": 0}

        def fake_generate(prompt_solutions, sampler_id):
            state["calls"] += 1
            if state["calls"] == 1:
                algo.run_state_dict.tot_sample_nums = 0
                raise RuntimeError("generation failed")
            return Solution("candidate"), {"sampler": sampler_id}

        def fake_evaluate(code: str):
            algo.run_state_dict.tot_sample_nums = cfg.max_sample_nums
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={})

        monkeypatch.setattr(algo, "_generate_single_program", fake_generate)
        monkeypatch.setattr(cfg.task, "evaluate_code", fake_evaluate)

        algo.run()

        assert algo.run_state_dict.is_done is True
