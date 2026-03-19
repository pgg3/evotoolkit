# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Additional branch coverage for control interfaces."""

import random

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.python_task.control_box2d import EoHControlInterface, EvoEngineerControlInterface, LunarLanderTask
from evotoolkit.task.python_task.control_box2d.method_interface.evoengineer_interface import (
    generate_episode_analysis,
    select_insights,
)


def _make_solution(score: float, *, success_rate: float = 0.5, name: str = "policy", thought: str = "steady") -> Solution:
    return Solution(
        "def policy(state):\n    return 0\n",
        other_info={"algorithm": "rule-based", "name": name, "thought": thought},
        evaluation_res=EvaluationResult(
            valid=True,
            score=score,
            additional_info={
                "success_rate": success_rate,
                "std_reward": 3.0,
                "min_reward": score - 20.0,
                "max_reward": score + 20.0,
                "avg_length": 180,
                "all_rewards": [score, score + 10.0, score - 100.0],
            },
        ),
    )


class TestEoHControlInterfaceExtended:
    def test_format_policy_info_without_optional_fields(self):
        task = LunarLanderTask(use_mock=True)
        iface = EoHControlInterface(task)
        solution = Solution("def policy(state):\n    return 0\n", other_info={}, evaluation_res=None)

        formatted = iface._format_policy_info(solution)

        assert "No description available" in formatted
        assert "Average Reward: 0.00" in formatted

    def test_all_prompt_operators_include_expected_sections(self):
        task = LunarLanderTask(use_mock=True)
        iface = EoHControlInterface(task)
        solution = _make_solution(120.0)

        e1 = iface.get_prompt_e1([solution])[0]["content"]
        e2 = iface.get_prompt_e2([solution])[0]["content"]
        m1 = iface.get_prompt_m1(solution)[0]["content"]
        m2 = iface.get_prompt_m2(solution)[0]["content"]

        assert "COMPLETELY DIFFERENT" in e1
        assert "COMMON BACKBONE IDEAS" in e2
        assert "OPTIMIZATION INSIGHT" in m1
        assert "TUNING THE PARAMETERS" in m2

    def test_parse_response_falls_back_to_raw_content(self):
        task = LunarLanderTask(use_mock=True)
        iface = EoHControlInterface(task)

        solution = iface.parse_response("No braces and no fenced code")

        assert solution.other_info["algorithm"] is None
        assert solution.sol_string == "No braces and no fenced code"


class TestEvoEngineerControlInterfaceExtended:
    def test_generate_episode_analysis_handles_missing_data(self):
        assert generate_episode_analysis(None) == "No episode data available."

    def test_select_insights_adds_performance_specific_guidance(self, monkeypatch):
        monkeypatch.setattr(random, "sample", lambda seq, n: list(seq)[:n])
        eval_result = EvaluationResult(
            valid=True,
            score=-10.0,
            additional_info={"success_rate": 0.1},
        )

        insights = select_insights(eval_result, n=3)

        assert any("survival" in insight for insight in insights)
        assert any("basic control" in insight for insight in insights)

    def test_select_insights_handles_high_success_rate_branch(self, monkeypatch):
        monkeypatch.setattr(random, "sample", lambda seq, n: list(seq)[:n])
        eval_result = EvaluationResult(
            valid=True,
            score=150.0,
            additional_info={"success_rate": 0.8},
        )

        insights = select_insights(eval_result, n=3)

        assert any("Both legs touching ground" in insight for insight in insights)

    def test_get_operator_prompt_supports_init_and_crossover(self):
        task = LunarLanderTask(use_mock=True)
        iface = EvoEngineerControlInterface(task)
        best = _make_solution(160.0, name="best", thought="balanced")
        parent = _make_solution(110.0, name="parent", thought="aggressive")

        init_prompt = iface.get_operator_prompt("init", [], best, [])[0]["content"]
        crossover_prompt = iface.get_operator_prompt("crossover", [parent, best], best, ["watch angle"])[0]["content"]

        assert "BASELINE POLICY" in init_prompt
        assert "PARENT POLICIES TO COMBINE" in crossover_prompt
        assert "watch angle" in crossover_prompt

    def test_get_operator_prompt_uses_make_init_sol_when_best_is_missing(self):
        task = LunarLanderTask(use_mock=True)
        iface = EvoEngineerControlInterface(task)
        target = _make_solution(100.0)

        prompt = iface.get_operator_prompt("mutation", [target], None, [])[0]["content"]

        assert "CONTROL POLICY MUTATION TASK" in prompt

    def test_parse_response_uses_fallback_paths(self):
        task = LunarLanderTask(use_mock=True)
        iface = EvoEngineerControlInterface(task)

        flexible = iface.parse_response("Name: flexible\nThought: ok\n```python\ndef policy(state):\n    return 1\n```")
        raw = iface.parse_response("Unstructured raw response")

        assert flexible.other_info["name"] == "flexible"
        assert "def policy" in flexible.sol_string
        assert raw.other_info["name"] == "raw"

    def test_extract_any_code_block_supports_plain_code_section(self):
        task = LunarLanderTask(use_mock=True)
        iface = EvoEngineerControlInterface(task)
        content = "code:\ndef policy(state):\n    return 2\nthought: concise"

        code = iface._extract_any_code_block(content)

        assert "def policy" in code
