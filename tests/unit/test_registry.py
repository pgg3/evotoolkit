# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for the registry system."""

import pytest

from evotoolkit.registry import (
    get_algorithm_info,
    get_task_class,
    infer_algorithm_from_interface,
    list_algorithms,
    list_tasks,
    register_task,
)


class TestTaskRegistry:
    def test_list_tasks_returns_list(self):
        tasks = list_tasks()
        assert isinstance(tasks, list)

    def test_list_tasks_includes_registered(self):
        # ScientificRegression is registered via import side-effect in evotoolkit
        import evotoolkit  # noqa: F401 - triggers registration

        tasks = list_tasks()
        assert len(tasks) > 0

    def test_get_task_class_invalid(self):
        with pytest.raises(ValueError, match="not found"):
            get_task_class("NonExistentTask_xyz_abc")

    def test_register_task_duplicate_raises(self):
        from evotoolkit.core.base_task import BaseTask
        from evotoolkit.core.solution import EvaluationResult, Solution

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
        assert "config" in info

    def test_get_algorithm_info_invalid(self):
        with pytest.raises(ValueError, match="not found"):
            get_algorithm_info("nonexistent_algo_xyz")


class TestInferAlgorithmFromInterface:
    def test_infer_eoh(self, minimal_task):
        from evotoolkit.task.python_task.method_interface import EoHPythonInterface

        iface = EoHPythonInterface(minimal_task)
        algo = infer_algorithm_from_interface(iface)
        assert algo == "eoh"

    def test_infer_evoengineer(self, minimal_task, evoengineer_interface):
        # Use a real EvoEngineer-prefixed interface; we can use EvoEngineerPythonInterface
        from evotoolkit.task.python_task.method_interface import EvoEngineerPythonInterface

        iface = EvoEngineerPythonInterface(minimal_task)
        algo = infer_algorithm_from_interface(iface)
        assert algo == "evoengineer"

    def test_infer_funsearch(self, minimal_task):
        from evotoolkit.task.python_task.method_interface import FunSearchPythonInterface

        iface = FunSearchPythonInterface(minimal_task)
        algo = infer_algorithm_from_interface(iface)
        assert algo == "funsearch"

    def test_infer_unknown_raises(self):
        from evotoolkit.core.method_interface import BaseMethodInterface
        from evotoolkit.core.solution import Solution

        class UnknownInterface(BaseMethodInterface):
            def make_init_sol(self):
                return Solution("code")

            def parse_response(self, response_str):
                return Solution(response_str)

        with pytest.raises(ValueError, match="Cannot infer algorithm"):
            infer_algorithm_from_interface(UnknownInterface(task=None))
