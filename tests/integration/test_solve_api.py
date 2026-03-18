# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Integration tests for evotoolkit.solve() end-to-end workflow using MockLLM."""

import evotoolkit
from evotoolkit.core import Solution
from evotoolkit.evo_method.eoh import EoH, EoHConfig
from evotoolkit.task.python_task.method_interface.eoh_interface import EoHPythonInterface

# ---------------------------------------------------------------------------
# MockLLM that returns a valid Python function
# ---------------------------------------------------------------------------

MOCK_CODE = "def f(x):\n    return x * 3"
MOCK_EoH_RESPONSE = "{A function that triples the input}\n```python\ndef f(x):\n    return x * 3\n```"


class MockLLM:
    """Deterministic mock LLM — always returns a fixed valid Python solution."""

    def __call__(self, messages, **kwargs):
        return {"content": MOCK_EoH_RESPONSE}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEoHIntegration:
    """Integration tests running EoH for 1 generation with MockLLM."""

    def test_eoh_config_creates(self, minimal_task, tmp_output):
        interface = EoHPythonInterface(minimal_task)
        cfg = EoHConfig(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        assert cfg.max_generations == 1
        assert cfg.pop_size == 2

    def test_eoh_initialises_run_state(self, minimal_task, tmp_output):
        interface = EoHPythonInterface(minimal_task)
        cfg = EoHConfig(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        algo = EoH(cfg)
        # run_state_dict should be initialized
        assert algo.run_state_dict is not None
        assert algo.run_state_dict.generation == 0

    def test_eoh_run_state_saved_to_file(self, minimal_task, tmp_output):
        import os

        interface = EoHPythonInterface(minimal_task)
        cfg = EoHConfig(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        EoH(cfg)
        # Constructor should save initial state
        assert os.path.exists(os.path.join(tmp_output, "run_state.json"))

    def test_solve_api_returns_solution(self, minimal_task, tmp_output):
        """Test the high-level evotoolkit.solve() API with MockLLM."""
        interface = EoHPythonInterface(minimal_task)
        result = evotoolkit.solve(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        assert isinstance(result, Solution)

    def test_solve_api_result_has_evaluation(self, minimal_task, tmp_output):
        interface = EoHPythonInterface(minimal_task)
        result = evotoolkit.solve(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        assert result.evaluation_res is not None
        assert result.evaluation_res.valid is True

    def test_solve_api_output_path_created(self, minimal_task, tmp_output):
        import os

        interface = EoHPythonInterface(minimal_task)
        evotoolkit.solve(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        assert os.path.isdir(tmp_output)
