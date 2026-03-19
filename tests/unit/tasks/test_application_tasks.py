# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for application-facing task implementations."""

import sys
import types

import numpy as np
import pytest

from evotoolkit.core import EvaluationResult
from evotoolkit.task.python_task.adversarial_attack import AdversarialAttackTask
from evotoolkit.task.python_task.control_box2d import LunarLanderTask
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.string_optimization.prompt_optimization import PromptOptimizationTask


class FakeLLM:
    def get_response(self, messages, **kwargs):
        return "The answer is 4", {}


class TestPromptOptimizationTask:
    def test_requires_llm_when_not_using_mock(self):
        with pytest.raises(ValueError, match="llm_api must be provided"):
            PromptOptimizationTask(test_cases=[{"question": "1+1", "expected": "2"}], use_mock=False)

    def test_mock_evaluation_scores_perfect_template(self):
        task = PromptOptimizationTask(
            test_cases=[
                {"question": "What is 2+2?", "expected": "4"},
                {"question": "What is 5*3?", "expected": "15"},
            ],
            use_mock=True,
        )

        result = task.evaluate_code("Solve this math problem: {question}\nGive only the number.")

        assert result.valid is True
        assert result.score == 1.0
        assert result.additional_info["correct"] == 2

    def test_invalid_template_is_rejected(self):
        task = PromptOptimizationTask(test_cases=[{"question": "What is 2+2?", "expected": "4"}], use_mock=True)

        result = task.evaluate_code("Answer directly")

        assert result.valid is False
        assert "placeholder" in result.additional_info["error"]

    def test_real_llm_path_uses_llm_api(self):
        task = PromptOptimizationTask(
            test_cases=[{"question": "What is 2+2?", "expected": "4"}],
            llm_api=FakeLLM(),
            use_mock=False,
        )

        result = task.evaluate_code("Question: {question}")

        assert result.valid is True
        assert result.score == 1.0

    def test_make_init_solution_is_evaluated(self):
        task = PromptOptimizationTask(test_cases=[{"question": "What is 2+2?", "expected": "4"}], use_mock=True)

        solution = task.make_init_sol_wo_other_info()

        assert solution.evaluation_res is not None
        assert solution.evaluation_res.valid is True


class TestScientificRegressionTask:
    @staticmethod
    def _mock_datasets():
        train_inputs = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        train_outputs = 2.0 * train_inputs[:, 0] + 3.0 * train_inputs[:, 1] + 1.0
        test_inputs = np.array([[4.0, 5.0], [5.0, 6.0]])
        test_outputs = 2.0 * test_inputs[:, 0] + 3.0 * test_inputs[:, 1] + 1.0
        return (
            {"inputs": train_inputs, "outputs": train_outputs},
            {"inputs": test_inputs, "outputs": test_outputs},
        )

    def test_valid_equation_evaluates_successfully(self, monkeypatch):
        datasets = self._mock_datasets()
        monkeypatch.setattr(ScientificRegressionTask, "_load_dataset", lambda self, dataset_name, data_dir: datasets)

        task = ScientificRegressionTask(dataset_name="oscillator1", max_params=3)
        result = task.evaluate_code(
            "import numpy as np\n"
            "def equation(x, v, params):\n"
            "    return params[0] * x + params[1] * v + params[2]\n"
        )

        assert task.task_info["dataset_name"] == "oscillator1"
        assert result.valid is True
        assert result.additional_info["train_mse"] >= 0.0
        assert result.additional_info["test_mse"] >= 0.0

    def test_missing_equation_is_invalid(self, monkeypatch):
        datasets = self._mock_datasets()
        monkeypatch.setattr(ScientificRegressionTask, "_load_dataset", lambda self, dataset_name, data_dir: datasets)
        task = ScientificRegressionTask(dataset_name="oscillator1", max_params=3)

        result = task.evaluate_code("def not_equation(x, v, params):\n    return x + v\n")

        assert result.valid is False
        assert "equation" in result.additional_info["error"]

    def test_make_init_solution_is_evaluated(self, monkeypatch):
        datasets = self._mock_datasets()
        monkeypatch.setattr(ScientificRegressionTask, "_load_dataset", lambda self, dataset_name, data_dir: datasets)
        task = ScientificRegressionTask(dataset_name="oscillator1", max_params=3)

        solution = task.make_init_sol_wo_other_info()

        assert solution.evaluation_res is not None
        assert solution.evaluation_res.valid is True


class TestAdversarialAttackTask:
    def test_mock_mode_returns_valid_result(self):
        task = AdversarialAttackTask(use_mock=True)

        result = task.evaluate_code("def anything():\n    return None\n")

        assert result.valid is True
        assert result.additional_info["mock"] is True

    def test_missing_draw_proposals_is_invalid(self):
        task = AdversarialAttackTask(use_mock=False)

        result = task.evaluate_code("x = 1\n")

        assert result.valid is False
        assert "draw_proposals" in result.additional_info["error"]

    def test_real_evaluation_path_uses_attack_score(self, monkeypatch):
        task = AdversarialAttackTask(use_mock=False, model=object(), test_loader=[])
        monkeypatch.setattr(task, "_evaluate_attack", lambda func: 1.5)

        result = task.evaluate_code(
            "def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):\n"
            "    return org_img\n"
        )

        assert result.valid is True
        assert result.score == -1.5
        assert result.additional_info["avg_distance"] == 1.5

    def test_nan_attack_score_is_rejected(self, monkeypatch):
        task = AdversarialAttackTask(use_mock=False, model=object(), test_loader=[])
        monkeypatch.setattr(task, "_evaluate_attack", lambda func: float("nan"))

        result = task.evaluate_code(
            "def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):\n"
            "    return org_img\n"
        )

        assert result.valid is False


class TestLunarLanderTask:
    def test_mock_mode_returns_valid_result(self):
        task = LunarLanderTask(use_mock=True, seed=0)

        result = task.evaluate_code("def policy(state):\n    return 0\n")

        assert result.valid is True
        assert result.additional_info["mock"] is True

    def test_missing_policy_is_invalid(self):
        task = LunarLanderTask(use_mock=False)

        result = task.evaluate_code("x = 1\n")

        assert result.valid is False
        assert "policy" in result.additional_info["error"]

    def test_real_evaluation_path_can_be_stubbed(self, monkeypatch):
        task = LunarLanderTask(use_mock=False)
        monkeypatch.setattr(task, "_evaluate_policy", lambda policy: EvaluationResult(valid=True, score=42.0, additional_info={"avg_reward": 42.0}))

        result = task.evaluate_code("def policy(state):\n    return 0\n")

        assert result.valid is True
        assert result.score == 42.0

    def test_fake_gym_environment_exercises_policy_loop(self, monkeypatch):
        class FakeEnv:
            def reset(self, seed=None):
                return np.zeros(8), {}

            def step(self, action):
                return np.zeros(8), 120.0, True, False, {}

            def close(self):
                return None

        fake_gym = types.SimpleNamespace(make=lambda *args, **kwargs: FakeEnv())
        monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)

        task = LunarLanderTask(num_episodes=2, max_steps=3, use_mock=False, seed=7)
        result = task._evaluate_policy(lambda state: 0)

        assert result.valid is True
        assert result.additional_info["success_rate"] == 1.0
        assert result.score == 120.0

    def test_invalid_action_type_is_reported(self, monkeypatch):
        class FakeEnv:
            def reset(self, seed=None):
                return np.zeros(8), {}

            def close(self):
                return None

        fake_gym = types.SimpleNamespace(make=lambda *args, **kwargs: FakeEnv())
        monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)

        task = LunarLanderTask(num_episodes=1, max_steps=1, use_mock=False)
        result = task._evaluate_policy(lambda state: "invalid")

        assert result.valid is False
        assert "Invalid action type" in result.additional_info["error"]
