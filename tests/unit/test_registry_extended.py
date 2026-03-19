# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Additional registry tests covering get_interface_class and get_task_class."""

import pytest

from evotoolkit.registry import (
    get_interface_class,
    get_task_class,
    list_tasks,
)


class TestGetTaskClass:
    def test_get_registered_task_class(self):
        import evotoolkit  # noqa: F401  trigger registration

        tasks = list_tasks()
        if tasks:
            task_name = tasks[0]
            cls = get_task_class(task_name)
            assert cls is not None

    def test_get_task_class_not_found_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_task_class("totally_nonexistent_task_xyzzy")


class TestGetInterfaceClass:
    def test_get_interface_class_eoh_python(self, minimal_task):
        iface_cls = get_interface_class("eoh", minimal_task)
        assert iface_cls.__name__ == "EoHPythonInterface"

    def test_get_interface_class_evoengineer_python(self, minimal_task):
        iface_cls = get_interface_class("evoengineer", minimal_task)
        assert iface_cls.__name__ == "EvoEngineerPythonInterface"

    def test_get_interface_class_funsearch_python(self, minimal_task):
        iface_cls = get_interface_class("funsearch", minimal_task)
        assert iface_cls.__name__ == "FunSearchPythonInterface"

    def test_get_interface_class_string_task(self):
        from evotoolkit.task.string_optimization import PromptOptimizationTask

        task = PromptOptimizationTask(test_cases=[{"question": "What is 2+2?", "expected": "4"}], use_mock=True)
        iface_cls = get_interface_class("eoh", task)

        assert iface_cls.__name__ == "EoHStringInterface"

    def test_get_interface_class_unknown_algorithm_raises(self, minimal_task):
        with pytest.raises(ValueError, match="No interface mapping"):
            get_interface_class("nonexistent_algo", minimal_task)

    def test_get_interface_class_unknown_task_type_raises(self, always_valid_task):
        """BaseTask that is not PythonTask and not CudaTask should raise ValueError."""
        with pytest.raises(ValueError):
            get_interface_class("eoh", always_valid_task)
