# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Execution-oriented tests for the built-in evolutionary algorithms."""

from pathlib import Path

from evotoolkit.core import Solution
from evotoolkit.evo_method.eoh import EoH
from evotoolkit.evo_method.evoengineer import EvoEngineer
from evotoolkit.evo_method.funsearch import FunSearch


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
        algo = EoH(
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

        result = algo.run()

        assert isinstance(result, Solution)
        assert algo.state.status == "completed"
        assert algo.state.tot_sample_nums > 0
        assert len(algo.state.population) <= algo.pop_size
        assert Path(tmp_output, "checkpoint", "state.pkl").exists()
        assert Path(tmp_output, "checkpoint", "manifest.json").exists()
        assert Path(tmp_output, "history").exists()

    def test_run_iteration_and_checkpoint_resume(self, eoh_interface, tmp_output):
        algo = EoH(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            verbose=False,
            max_generations=2,
            max_sample_nums=6,
            pop_size=2,
            selection_num=1,
            use_e2_operator=False,
            use_m1_operator=True,
            use_m2_operator=False,
            num_samplers=2,
            num_evaluators=2,
        )
        algo.run_iteration()

        restored = EoH(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            verbose=False,
            max_generations=3,
            max_sample_nums=8,
            pop_size=2,
            selection_num=1,
            use_e2_operator=False,
            use_m1_operator=True,
            use_m2_operator=False,
            num_samplers=2,
            num_evaluators=2,
        )
        restored.load_checkpoint()

        assert restored.state.tot_sample_nums == algo.state.tot_sample_nums
        assert restored.state.generation == algo.state.generation

    def test_generate_single_initial_solution_handles_llm_errors(self, eoh_interface, tmp_output):
        algo = EoH(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=FailingLLM(),
            verbose=False,
            num_samplers=1,
            num_evaluators=1,
        )
        solution, usage = algo._generate_single_initial_solution(0)

        assert solution.sol_string == ""
        assert usage == {}


class TestEvoEngineerExecution:
    def test_run_completes_and_keeps_population_bounded(self, evoengineer_interface, tmp_output):
        algo = EvoEngineer(
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

        result = algo.run()

        assert isinstance(result, Solution)
        assert algo.state.status == "completed"
        assert algo.state.tot_sample_nums > 0
        assert len(algo.state.population) <= algo.pop_size
        assert len(algo.state.usage_history["sample"]) > 0
        assert Path(tmp_output, "checkpoint", "state.pkl").exists()

    def test_load_checkpoint_restores_generation_state(self, evoengineer_interface, tmp_output):
        algo = EvoEngineer(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            verbose=False,
            max_generations=2,
            max_sample_nums=6,
            pop_size=3,
            num_samplers=4,
            num_evaluators=2,
        )
        algo.run_iteration()

        restored = EvoEngineer(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            verbose=False,
            max_generations=3,
            max_sample_nums=8,
            pop_size=3,
            num_samplers=4,
            num_evaluators=2,
        )
        restored.load_checkpoint()

        assert restored.state.generation == algo.state.generation
        assert restored.state.tot_sample_nums == algo.state.tot_sample_nums

    def test_generate_single_solution_handles_llm_errors(self, evoengineer_interface, tmp_output):
        algo = EvoEngineer(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=FailingLLM(),
            verbose=False,
            num_samplers=2,
            num_evaluators=1,
        )
        operator = algo.init_operators[0]
        solution, usage = algo._generate_single_solution(operator, [], 0)

        assert solution.sol_string == ""
        assert usage == {}


class TestFunSearchExecution:
    def test_run_saves_checkpoint_without_external_database_file(self, funsearch_interface, tmp_output):
        algo = FunSearch(
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

        result = algo.run()

        assert isinstance(result, Solution)
        assert algo.state.status == "completed"
        assert algo.state.tot_sample_nums == 4
        assert algo.state.programs_database is not None
        assert Path(tmp_output, "checkpoint", "state.pkl").exists()
        assert not Path(tmp_output, "programs_database.json").exists()
        assert Path(tmp_output, "history", "batch_0000.json").exists()

    def test_run_can_resume_from_checkpoint(self, funsearch_interface, tmp_output):
        algo = FunSearch(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            verbose=False,
            max_sample_nums=2,
            num_islands=2,
            num_samplers=2,
            num_evaluators=2,
            programs_per_prompt=1,
        )
        algo.run()

        restored = FunSearch(
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
        restored.load_checkpoint()
        restored.run()

        assert restored.state.programs_database is not None
        assert restored.state.tot_sample_nums == 4

    def test_generate_single_program_handles_llm_errors(self, funsearch_interface, tmp_output):
        algo = FunSearch(
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
        prompt_solutions = [funsearch_interface.task.make_init_sol_wo_other_info()]

        solution, usage = algo._generate_single_program(prompt_solutions, 0)

        assert solution.sol_string == ""
        assert usage == {}
