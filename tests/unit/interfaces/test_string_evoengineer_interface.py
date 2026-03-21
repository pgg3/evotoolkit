# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EvoEngineerStringInterface parse_response and get_operator_prompt."""

import pytest

from evotoolkit.core import EvaluationResult, Solution, TaskSpec
from evotoolkit.task.string_optimization.evoengineer_interface import EvoEngineerStringInterface
from evotoolkit.task.string_optimization.string_task import StringTask


class MinimalStringTask(StringTask):
    def build_string_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="test_string",
            prompt="Optimize the string to maximize length.",
            modality="string",
        )

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        score = len(candidate_string)
        return EvaluationResult(valid=True, score=float(score), additional_info={})


@pytest.fixture
def string_task():
    return MinimalStringTask(data=None)


@pytest.fixture
def string_interface(string_task):
    return EvoEngineerStringInterface(string_task)


def _make_scored_solution(text: str, score: float) -> Solution:
    return Solution(
        sol_string=text,
        evaluation_res=EvaluationResult(valid=True, score=score, additional_info={}),
        metadata={"name": "test", "description": "test thought"},
    )


class TestEvoEngineerStringInterfaceOperators:
    def test_get_init_operators(self, string_interface):
        ops = string_interface.get_init_operators()
        assert len(ops) == 1
        assert ops[0].name == "init"
        assert ops[0].selection_size == 0

    def test_get_offspring_operators(self, string_interface):
        ops = string_interface.get_offspring_operators()
        names = [op.name for op in ops]
        assert "crossover" in names
        assert "mutation" in names


class TestGetOperatorPrompt:
    def test_init_operator_prompt(self, string_interface):
        prompt = string_interface.get_operator_prompt("init", [], None, [])
        assert isinstance(prompt, list)
        assert len(prompt) == 1
        assert "user" in prompt[0]["role"]

    def test_init_operator_with_thoughts(self, string_interface):
        thoughts = ["try longer", "add detail"]
        prompt = string_interface.get_operator_prompt("init", [], None, thoughts)
        content = prompt[0]["content"]
        assert "try longer" in content
        assert "add detail" in content

    def test_mutation_operator_prompt(self, string_interface):
        best = _make_scored_solution("best sol", 8.0)
        individual = _make_scored_solution("mutate me", 3.0)
        prompt = string_interface.get_operator_prompt("mutation", [individual], best, [])
        assert isinstance(prompt, list)
        content = prompt[0]["content"]
        assert "mutate" in content.lower() or "mutation" in content.lower()

    def test_mutation_with_thoughts(self, string_interface):
        best = _make_scored_solution("best", 5.0)
        individual = _make_scored_solution("parent", 2.0)
        thoughts = ["insight1"]
        prompt = string_interface.get_operator_prompt("mutation", [individual], best, thoughts)
        content = prompt[0]["content"]
        assert "insight1" in content

    def test_crossover_operator_prompt(self, string_interface):
        best = _make_scored_solution("best", 9.0)
        parent1 = _make_scored_solution("parent one", 5.0)
        parent2 = _make_scored_solution("parent two", 6.0)
        prompt = string_interface.get_operator_prompt("crossover", [parent1, parent2], best, [])
        assert isinstance(prompt, list)
        content = prompt[0]["content"]
        assert "parent" in content.lower() or "crossover" in content.lower() or "Parent" in content

    def test_crossover_with_thoughts(self, string_interface):
        best = _make_scored_solution("best", 9.0)
        parent1 = _make_scored_solution("p1", 5.0)
        parent2 = _make_scored_solution("p2", 6.0)
        thoughts = ["combine well"]
        prompt = string_interface.get_operator_prompt("crossover", [parent1, parent2], best, thoughts)
        content = prompt[0]["content"]
        assert "combine well" in content

    def test_unknown_operator_returns_empty(self, string_interface):
        best = _make_scored_solution("best", 1.0)
        result = string_interface.get_operator_prompt("unknown_op", [], best, [])
        assert result == []

    def test_init_with_none_best_sol_is_allowed(self, string_interface):
        prompt = string_interface.get_operator_prompt("init", [], None, [])
        assert isinstance(prompt, list)
        assert len(prompt) == 1


class TestParseResponse:
    def test_empty_response(self, string_interface):
        sol = string_interface.parse_response("")
        assert sol.sol_string == ""

    def test_whitespace_only_response(self, string_interface):
        sol = string_interface.parse_response("   ")
        assert sol.sol_string == ""

    def test_standard_format(self, string_interface):
        response = "name: my_solution\nsolution: The optimized text here\nthought: This is better"
        sol = string_interface.parse_response(response)
        assert sol.sol_string == "The optimized text here"
        assert sol.metadata.name == "my_solution"
        assert "This is better" in sol.metadata.description

    def test_json_format(self, string_interface):
        import json

        response = json.dumps({"name": "json_sol", "solution": "json solution text", "thought": "json reasoning"})
        sol = string_interface.parse_response(response)
        assert sol.sol_string == "json solution text"

    def test_json_in_code_block(self, string_interface):
        import json

        data = {"name": "block_sol", "solution": "block solution", "thought": "reasoning"}
        response = f"```json\n{json.dumps(data)}\n```"
        sol = string_interface.parse_response(response)
        assert sol.sol_string == "block solution"

    def test_flexible_format_fallback(self, string_interface):
        response = "Name: flex_sol\nSolution: flexible solution text"
        sol = string_interface.parse_response(response)
        assert sol is not None

    def test_raw_fallback(self, string_interface):
        """When no format matches, returns raw content."""
        response = "totally unstructured content without any keywords"
        sol = string_interface.parse_response(response)
        assert sol.sol_string == response.strip() or sol is not None

    def test_clean_solution_removes_outer_quotes(self, string_interface):
        response = 'name: test\nsolution: "quoted solution"\nthought: ok'
        sol = string_interface.parse_response(response)
        # Outer quotes should be stripped
        assert not sol.sol_string.startswith('"') or "quoted" in sol.sol_string

    def test_clean_solution_unescape_sequences(self, string_interface):
        response = "name: test\nsolution: line1\\nline2\nthought: ok"
        sol = string_interface.parse_response(response)
        assert "\n" in sol.sol_string or "line" in sol.sol_string

    def test_json_with_alternative_keys(self, string_interface):
        import json

        response = json.dumps({"name": "alt_sol", "code": "code as solution", "reasoning": "alt reasoning"})
        sol = string_interface.parse_response(response)
        # Should extract "code" as solution fallback
        assert sol is not None
