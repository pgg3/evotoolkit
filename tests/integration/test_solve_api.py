# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Integration tests for the explicit low-level runtime API."""

from evotoolkit.core import Solution
from evotoolkit.evo_method.eoh import EoH
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

    def get_response(self, messages, **kwargs):
        return MOCK_EoH_RESPONSE, {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEoHIntegration:
    """Integration tests running EoH explicitly for 1 generation with MockLLM."""

    def test_eoh_initialises_state(self, minimal_task, tmp_output):
        interface = EoHPythonInterface(minimal_task)
        algo = EoH(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        assert algo.state is not None
        assert algo.state.generation == 0

    def test_eoh_checkpoint_saved_to_file(self, minimal_task, tmp_output):
        import os

        interface = EoHPythonInterface(minimal_task)
        algo = EoH(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        algo.run_iteration()
        assert os.path.exists(os.path.join(tmp_output, "checkpoint", "state.pkl"))

    def test_explicit_run_returns_solution(self, minimal_task, tmp_output):
        interface = EoHPythonInterface(minimal_task)
        algo = EoH(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        result = algo.run()
        assert isinstance(result, Solution)
        assert result.evaluation_res is not None
        assert result.evaluation_res.valid is True

    def test_explicit_run_output_path_created(self, minimal_task, tmp_output):
        import os

        interface = EoHPythonInterface(minimal_task)
        algo = EoH(
            interface=interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
            max_generations=1,
            pop_size=2,
            verbose=False,
        )
        algo.run()
        assert os.path.isdir(tmp_output)
