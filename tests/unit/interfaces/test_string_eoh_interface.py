# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EoHStringInterface prompt generation and parsing."""

import re

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.string_optimization.method_interface.eoh_interface import EoHStringInterface
from evotoolkit.task.string_optimization.string_task import StringTask


class MinimalStringTask(StringTask):
    """Minimal concrete StringTask for testing."""

    def _process_data(self, data):
        self.data = data
        self.task_info = {"name": "string_test"}

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        return EvaluationResult(valid=True, score=float(len(candidate_string)), additional_info={})

    def get_base_task_description(self) -> str:
        return "Produce a long and interesting string."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("hello world")


@pytest.fixture
def string_task():
    return MinimalStringTask(data=None)


class ConcreteEoHStringInterface(EoHStringInterface):
    """Concrete subclass implementing remaining abstract methods."""

    def get_prompt_m2(self, individual: Solution) -> list:
        return [{"role": "user", "content": f"Perturb: {individual.sol_string}"}]

    def parse_response(self, response_str: str) -> Solution:
        # Extract content inside {{ }} as algorithm, rest as solution
        algo_match = re.search(r"\{([^}]+)\}", response_str)
        algo = algo_match.group(1).strip() if algo_match else ""
        sol_string = re.sub(r"\{[^}]*\}", "", response_str).strip()
        return Solution(sol_string or response_str.strip(), other_info={"algorithm": algo})


@pytest.fixture
def string_iface(string_task):
    return ConcreteEoHStringInterface(string_task)


@pytest.fixture
def sol_with_algorithm():
    return Solution(
        sol_string="The quick brown fox",
        other_info={"algorithm": "Use vivid adjectives"},
        evaluation_res=EvaluationResult(valid=True, score=19.0, additional_info={}),
    )


@pytest.fixture
def sol_without_algorithm():
    return Solution(
        sol_string="Hello world",
        other_info={},
        evaluation_res=EvaluationResult(valid=True, score=11.0, additional_info={}),
    )


class TestEoHStringInterfaceInit:
    def test_interface_creates(self, string_iface):
        assert string_iface is not None
        assert string_iface.task is not None


class TestEoHStringInterfacePromptI1:
    def test_returns_message_list(self, string_iface):
        msgs = string_iface.get_prompt_i1()
        assert isinstance(msgs, list)
        assert len(msgs) == 1

    def test_message_has_user_role(self, string_iface):
        msgs = string_iface.get_prompt_i1()
        assert msgs[0]["role"] == "user"

    def test_contains_task_description(self, string_iface):
        msgs = string_iface.get_prompt_i1()
        assert "string" in msgs[0]["content"].lower() or "long" in msgs[0]["content"].lower()


class TestEoHStringInterfacePromptE1:
    def test_returns_message_list(self, string_iface, sol_with_algorithm):
        msgs = string_iface.get_prompt_e1([sol_with_algorithm])
        assert isinstance(msgs, list)
        assert len(msgs) == 1

    def test_includes_solution_string(self, string_iface, sol_with_algorithm):
        msgs = string_iface.get_prompt_e1([sol_with_algorithm])
        content = msgs[0]["content"]
        assert sol_with_algorithm.sol_string in content

    def test_includes_algorithm_description(self, string_iface, sol_with_algorithm):
        msgs = string_iface.get_prompt_e1([sol_with_algorithm])
        content = msgs[0]["content"]
        assert "vivid adjectives" in content

    def test_fallback_when_no_algorithm(self, string_iface, sol_without_algorithm):
        msgs = string_iface.get_prompt_e1([sol_without_algorithm])
        content = msgs[0]["content"]
        assert "Solution 1" in content

    def test_multiple_individuals(self, string_iface, sol_with_algorithm, sol_without_algorithm):
        msgs = string_iface.get_prompt_e1([sol_with_algorithm, sol_without_algorithm])
        content = msgs[0]["content"]
        assert "No. 1" in content
        assert "No. 2" in content


class TestEoHStringInterfacePromptE2:
    def test_returns_message_list(self, string_iface, sol_with_algorithm):
        msgs = string_iface.get_prompt_e2([sol_with_algorithm])
        assert isinstance(msgs, list)

    def test_mentions_backbone(self, string_iface, sol_with_algorithm):
        msgs = string_iface.get_prompt_e2([sol_with_algorithm, sol_with_algorithm])
        content = msgs[0]["content"]
        assert "backbone" in content.lower() or "common" in content.lower()


class TestEoHStringInterfacePromptM1:
    def test_returns_message_list(self, string_iface, sol_with_algorithm):
        msgs = string_iface.get_prompt_m1(sol_with_algorithm)
        assert isinstance(msgs, list)
        assert len(msgs) == 1

    def test_includes_individual_string(self, string_iface, sol_with_algorithm):
        msgs = string_iface.get_prompt_m1(sol_with_algorithm)
        content = msgs[0]["content"]
        assert sol_with_algorithm.sol_string in content

    def test_fallback_when_no_algorithm(self, string_iface, sol_without_algorithm):
        msgs = string_iface.get_prompt_m1(sol_without_algorithm)
        content = msgs[0]["content"]
        assert "Current solution" in content
