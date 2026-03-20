# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for the registry system."""

import pytest

from evotoolkit.core.base_task import BaseTask
from evotoolkit.core.solution import EvaluationResult, Solution
from evotoolkit.registry import (
    get_algorithm_info,
    get_task_class,
    list_algorithms,
    list_tasks,
    register_task,
)


class TestTaskRegistry:
    def test_list_tasks_returns_list(self):
        tasks = list_tasks()
        assert isinstance(tasks, list)

    def test_list_tasks_includes_registered(self):
        unique_name = "_TestRegisteredTask_" + str(id(object()))

        @register_task(unique_name)
        class TmpTask(BaseTask):
            def evaluate_code(self, code):
                return EvaluationResult(True, 0.0, {})

            def get_base_task_description(self):
                return "tmp"

            def make_init_sol_wo_other_info(self):
                return Solution("code")

        tasks = list_tasks()
        assert unique_name in tasks

    def test_get_task_class_invalid(self):
        with pytest.raises(ValueError, match="not found"):
            get_task_class("NonExistentTask_xyz_abc")

    def test_register_task_duplicate_raises(self):
        # Use a unique name to avoid conflicts
        unique_name = "_TestDuplicateTask_" + str(id(object()))

        @register_task(unique_name)
        class TmpTask(BaseTask):
            def evaluate_code(self, code):
                return EvaluationResult(True, 0.0, {})

            def get_base_task_description(self):
                return "tmp"

            def make_init_sol_wo_other_info(self):
                return Solution("code")

        with pytest.raises(ValueError, match="already registered"):

            @register_task(unique_name)
            class TmpTask2(BaseTask):
                def evaluate_code(self, code):
                    return EvaluationResult(True, 0.0, {})

                def get_base_task_description(self):
                    return "tmp2"

                def make_init_sol_wo_other_info(self):
                    return Solution("code")


class TestAlgorithmRegistry:
    def test_list_algorithms_returns_list(self):
        import evotoolkit  # noqa: F401 - triggers algorithm registration

        algos = list_algorithms()
        assert isinstance(algos, list)
        assert len(algos) > 0

    def test_list_algorithms_contains_known(self):
        import evotoolkit  # noqa: F401

        algos = list_algorithms()
        assert "eoh" in algos
        assert "evoengineer" in algos
        assert "funsearch" in algos

    def test_get_algorithm_info_eoh(self):
        import evotoolkit  # noqa: F401

        info = get_algorithm_info("eoh")
        assert "class" in info
        assert info["class"].__name__ == "EoH"

    def test_get_algorithm_info_invalid(self):
        with pytest.raises(ValueError, match="not found"):
            get_algorithm_info("nonexistent_algo_xyz")
