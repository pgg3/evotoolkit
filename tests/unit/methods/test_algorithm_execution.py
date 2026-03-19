# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Execution-oriented tests for the built-in evolutionary algorithms."""

from pathlib import Path

from evotoolkit.evo_method.eoh import EoH, EoHConfig
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig
from evotoolkit.evo_method.funsearch import FunSearch, FunSearchConfig


class MockLLM:
    """Deterministic mock LLM used by algorithm execution tests."""

    def __init__(self, response: str = "mock response"):
        self.response = response
        self.calls = []

    def get_response(self, messages, **kwargs):
        self.calls.append(messages)
        return self.response, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class FailingLLM:
    """Mock LLM that always raises."""

    def get_response(self, messages, **kwargs):
        raise RuntimeError("mock llm failure")


class TestEoHExecution:
    def test_run_completes_and_saves_state(self, eoh_interface, tmp_output):
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            verbose=False,
            max_generations=2,
            max_sample_nums=8,
            pop_size=2,
            selection_num=1,
            use_e2_operator=False,
            use_m1_operator=True,
            use_m2_operator=False,
            num_samplers=2,
            num_evaluators=2,
        )

        algo = EoH(cfg)
        algo.run()

        assert algo.run_state_dict.is_done is True
        assert algo.run_state_dict.tot_sample_nums > 0
        assert len(algo.run_state_dict.population) <= cfg.pop_size
        assert Path(tmp_output, "run_state.json").exists()

    def test_generate_single_initial_solution_handles_llm_errors(self, eoh_interface, tmp_output):
        cfg = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=FailingLLM(),
            verbose=False,
            num_samplers=1,
            num_evaluators=1,
        )

        algo = EoH(cfg)
        solution, usage = algo._generate_single_initial_solution(0)

        assert solution.sol_string == ""
        assert usage == {}


class TestEvoEngineerExecution:
    def test_run_completes_and_keeps_population_bounded(self, evoengineer_interface, tmp_output):
        cfg = EvoEngineerConfig(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            verbose=False,
            max_generations=2,
            max_sample_nums=10,
            pop_size=3,
            num_samplers=4,
            num_evaluators=2,
        )

        algo = EvoEngineer(cfg)
        algo.run()

        assert algo.run_state_dict.is_done is True
        assert algo.run_state_dict.tot_sample_nums > 0
        assert len(algo.run_state_dict.population) <= cfg.pop_size
        assert len(algo.run_state_dict.usage_history["sample"]) > 0

    def test_generate_single_solution_handles_llm_errors(self, evoengineer_interface, tmp_output):
        cfg = EvoEngineerConfig(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=FailingLLM(),
            verbose=False,
            num_samplers=2,
            num_evaluators=1,
        )

        algo = EvoEngineer(cfg)
        operator = cfg.get_init_operators()[0]
        solution, usage = algo._generate_single_solution(operator, [], 0)

        assert solution.sol_string == ""
        assert usage == {}


class TestFunSearchExecution:
    def test_run_saves_database_and_can_restore(self, funsearch_interface, tmp_output):
        cfg = FunSearchConfig(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            verbose=False,
            max_sample_nums=4,
            num_islands=2,
            num_samplers=2,
            num_evaluators=2,
            programs_per_prompt=1,
        )

        algo = FunSearch(cfg)
        algo.run()

        assert algo.run_state_dict.is_done is True
        assert algo.run_state_dict.tot_sample_nums == 4
        assert Path(algo.run_state_dict.database_file).exists()
        assert Path(tmp_output, "run_state.json").exists()

        restored = FunSearch(cfg)
        restored.run()

        assert restored.run_state_dict.is_done is True
        assert restored.run_state_dict.database_file == algo.run_state_dict.database_file
        assert restored.run_state_dict.tot_sample_nums == algo.run_state_dict.tot_sample_nums

    def test_generate_single_program_handles_llm_errors(self, funsearch_interface, tmp_output):
        cfg = FunSearchConfig(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=FailingLLM(),
            verbose=False,
            max_sample_nums=2,
            num_islands=1,
            num_samplers=1,
            num_evaluators=1,
            programs_per_prompt=1,
        )

        algo = FunSearch(cfg)
        prompt_solutions = [funsearch_interface.make_init_sol()]

        solution, usage = algo._generate_single_program(prompt_solutions, 0)

        assert solution.sol_string == ""
        assert usage == {}
