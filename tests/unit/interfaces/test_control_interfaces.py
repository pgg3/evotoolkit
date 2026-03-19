# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for control-task method interfaces."""

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.python_task.control_box2d import (
    EoHControlInterface,
    EvoEngineerControlInterface,
    FunSearchControlInterface,
    LunarLanderTask,
)


def _make_solution(score: float, name: str = "policy", thought: str = "rule-based") -> Solution:
    return Solution(
        "def policy(state):\n    return 0\n",
        other_info={"algorithm": "keep stable", "name": name, "thought": thought},
        evaluation_res=EvaluationResult(
            valid=True,
            score=score,
            additional_info={
                "success_rate": 0.5,
                "std_reward": 10.0,
                "min_reward": score - 10.0,
                "max_reward": score + 10.0,
                "avg_length": 200.0,
                "all_rewards": [score, score + 5.0],
            },
        ),
    )


class TestEoHControlInterface:
    def test_prompt_i1_contains_task_description(self):
        task = LunarLanderTask(use_mock=True)
        iface = EoHControlInterface(task)

        prompt = iface.get_prompt_i1()

        assert "CONTROL POLICY EVOLUTION TASK" in prompt[0]["content"]
        assert "policy(state" in prompt[0]["content"]

    def test_parse_response_extracts_algorithm_and_code(self):
        task = LunarLanderTask(use_mock=True)
        iface = EoHControlInterface(task)

        solution = iface.parse_response("{Try a stable hover controller}\n```python\ndef policy(state):\n    return 0\n```")

        assert solution.other_info["algorithm"] == "Try a stable hover controller"
        assert "def policy" in solution.sol_string


class TestEvoEngineerControlInterface:
    def test_operator_sets_are_exposed(self):
        task = LunarLanderTask(use_mock=True)
        iface = EvoEngineerControlInterface(task)

        assert [op.name for op in iface.get_init_operators()] == ["init"]
        assert [op.name for op in iface.get_offspring_operators()] == ["crossover", "mutation"]

    def test_get_operator_prompt_returns_structured_prompt(self):
        task = LunarLanderTask(use_mock=True)
        iface = EvoEngineerControlInterface(task)
        best = _make_solution(150.0, name="best", thought="balanced landing")

        prompt = iface.get_operator_prompt("mutation", [best], best, [])

        assert "CONTROL POLICY MUTATION TASK" in prompt[0]["content"]
        assert "policy(state" in prompt[0]["content"]

    def test_parse_response_extracts_name_code_and_thought(self):
        task = LunarLanderTask(use_mock=True)
        iface = EvoEngineerControlInterface(task)

        solution = iface.parse_response(
            "name: smoother_controller\n"
            "code:\n```python\ndef policy(state):\n    return 0\n```\n"
            "thought: reduced oscillation by simplifying thresholds"
        )

        assert solution.other_info["name"] == "smoother_controller"
        assert solution.other_info["thought"] == "reduced oscillation by simplifying thresholds"
        assert "def policy" in solution.sol_string


class TestFunSearchControlInterface:
    def test_single_solution_prompt(self):
        task = LunarLanderTask(use_mock=True)
        iface = FunSearchControlInterface(task)
        prompt = iface.get_prompt([_make_solution(120.0)])

        assert "CURRENT POLICY" in prompt[0]["content"]

    def test_progression_prompt(self):
        task = LunarLanderTask(use_mock=True)
        iface = FunSearchControlInterface(task)
        worse = _make_solution(80.0, name="worse")
        better = _make_solution(140.0, name="better")

        prompt = iface.get_prompt([worse, better])

        assert "POLICY PROGRESSION" in prompt[0]["content"]
        assert "Previous Policy" in prompt[0]["content"]

    def test_parse_response_extracts_code_block(self):
        task = LunarLanderTask(use_mock=True)
        iface = FunSearchControlInterface(task)

        solution = iface.parse_response("```python\ndef policy(state):\n    return 0\n```")

        assert solution.sol_string.strip().startswith("def policy")
