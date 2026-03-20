# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EoHPythonInterface."""

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.python_task import EoHPythonInterface


def _make_initial_solution(interface) -> Solution:
    return interface.make_solution(
        interface.task.spec.initial_solution,
        name=interface.task.spec.initial_name,
        description=interface.task.spec.initial_description,
        extras=interface.task.spec.initial_extras,
    )


class TestEoHPythonInterfaceTaskInitSolution:
    def test_task_initial_solution_returns_solution(self, eoh_interface):
        sol = _make_initial_solution(eoh_interface)
        assert isinstance(sol, Solution)

    def test_task_initial_solution_has_no_algorithm_metadata(self, eoh_interface):
        sol = _make_initial_solution(eoh_interface)
        assert sol.metadata.description == ""


class TestEoHPythonInterfacePrompts:
    """Test that prompts are non-empty lists of message dicts."""

    def _assert_prompt_valid(self, prompt):
        assert isinstance(prompt, list)
        assert len(prompt) > 0
        for msg in prompt:
            assert "role" in msg
            assert "content" in msg
            assert len(msg["content"]) > 0

    def test_get_prompt_i1(self, eoh_interface):
        prompt = eoh_interface.get_prompt_i1()
        self._assert_prompt_valid(prompt)

    def test_get_prompt_e1(self, eoh_interface, valid_solution):
        sol = _make_initial_solution(eoh_interface)
        prompt = eoh_interface.get_prompt_e1([sol])
        self._assert_prompt_valid(prompt)

    def test_get_prompt_e2(self, eoh_interface, valid_solution):
        sol = _make_initial_solution(eoh_interface)
        prompt = eoh_interface.get_prompt_e2([sol, sol])
        self._assert_prompt_valid(prompt)

    def test_get_prompt_m1(self, eoh_interface):
        sol = _make_initial_solution(eoh_interface)
        prompt = eoh_interface.get_prompt_m1(sol)
        self._assert_prompt_valid(prompt)

    def test_get_prompt_m2(self, eoh_interface):
        sol = _make_initial_solution(eoh_interface)
        prompt = eoh_interface.get_prompt_m2(sol)
        self._assert_prompt_valid(prompt)

    def test_i1_contains_task_description(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        prompt = iface.get_prompt_i1()
        content = prompt[0]["content"]
        assert minimal_task.spec.prompt in content

    def test_e1_prompt_mentions_individuals(self, minimal_task):
        iface = EoHPythonInterface(minimal_task)
        sol = _make_initial_solution(iface)
        prompt = iface.get_prompt_e1([sol, sol])
        content = prompt[0]["content"]
        assert "2" in content  # mentions count of individuals

    def test_prompts_tolerate_missing_metadata(self, eoh_interface):
        sol = Solution(
            "def f(x):\n    return x",
            evaluation_res=EvaluationResult(valid=True, score=1.0, additional_info={}),
        )
        prompt = eoh_interface.get_prompt_m1(sol)
        self._assert_prompt_valid(prompt)


class TestEoHPythonInterfaceParseResponse:
    @pytest.fixture
    def real_eoh_interface(self, minimal_task):
        return EoHPythonInterface(minimal_task)

    def test_parse_code_block(self, real_eoh_interface):
        response = "{Linear function}\n```python\ndef f(x):\n    return x\n```"
        sol = real_eoh_interface.parse_response(response)
        assert isinstance(sol, Solution)
        assert "def f(x)" in sol.sol_string

    def test_parse_extracts_algorithm_desc(self, real_eoh_interface):
        response = "{This is the algorithm description}\n```python\ndef f(x): return x\n```"
        sol = real_eoh_interface.parse_response(response)
        assert "algorithm description" in sol.metadata.description

    def test_parse_no_code_block_returns_string(self, real_eoh_interface):
        response = "def f(x): return x"
        sol = real_eoh_interface.parse_response(response)
        assert isinstance(sol, Solution)
        assert sol.sol_string  # non-empty

    def test_parse_empty_response(self, real_eoh_interface):
        sol = real_eoh_interface.parse_response("")
        assert isinstance(sol, Solution)

    def test_parse_no_algorithm_desc(self, real_eoh_interface):
        response = "```python\ndef f(x): return x\n```"
        sol = real_eoh_interface.parse_response(response)
        assert sol.metadata.description == ""
