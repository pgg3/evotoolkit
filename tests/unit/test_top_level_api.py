# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for evotoolkit top-level API and version info."""

import evotoolkit
import evotoolkit.core as runtime_core


class TestTopLevelAPI:
    def test_version_string_exists(self):
        assert hasattr(evotoolkit, "__version__")
        assert isinstance(evotoolkit.__version__, str)
        assert len(evotoolkit.__version__) > 0

    def test_author_string_exists(self):
        assert hasattr(evotoolkit, "__author__")
        assert isinstance(evotoolkit.__author__, str)

    def test_list_algorithms_callable(self):
        algos = evotoolkit.list_algorithms()
        assert algos == ["eoh", "evoengineer", "funsearch"]

    def test_list_tasks_not_exported_at_top_level(self):
        assert not hasattr(evotoolkit, "list_tasks")

    def test_solve_is_not_exported(self):
        assert not hasattr(evotoolkit, "solve")

    def test_algorithm_classes_are_exported(self):
        assert evotoolkit.EoH.__name__ == "EoH"
        assert evotoolkit.EvoEngineer.__name__ == "EvoEngineer"
        assert evotoolkit.FunSearch.__name__ == "FunSearch"

    def test_all_exports(self):
        for name in evotoolkit.__all__:
            assert hasattr(evotoolkit, name), f"Missing export: {name}"

    def test_core_exports_only_neutral_runtime_types(self):
        expected_exports = {
            "Solution",
            "SolutionMetadata",
            "EvaluationResult",
            "Task",
            "TaskSpec",
            "Method",
            "IterativeMethod",
            "PopulationMethod",
            "MethodInterface",
            "MethodState",
            "IterationState",
            "PopulationState",
            "RunStore",
        }
        assert set(runtime_core.__all__) == expected_exports
        assert not hasattr(runtime_core, "Operator")
        assert not hasattr(runtime_core, "EoHInterface")
        assert not hasattr(runtime_core, "EvoEngineerInterface")
        assert not hasattr(runtime_core, "FunSearchInterface")
