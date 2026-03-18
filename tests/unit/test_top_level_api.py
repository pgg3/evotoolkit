# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for evotoolkit top-level API and version info."""

import evotoolkit


class TestTopLevelAPI:
    def test_version_string_exists(self):
        assert hasattr(evotoolkit, "__version__")
        assert isinstance(evotoolkit.__version__, str)
        assert len(evotoolkit.__version__) > 0

    def test_author_string_exists(self):
        assert hasattr(evotoolkit, "__author__")
        assert isinstance(evotoolkit.__author__, str)

    def test_solve_is_callable(self):
        assert callable(evotoolkit.solve)

    def test_list_tasks_callable(self):
        tasks = evotoolkit.list_tasks()
        assert isinstance(tasks, list)

    def test_list_algorithms_callable(self):
        algos = evotoolkit.list_algorithms()
        assert isinstance(algos, list)
        assert "eoh" in algos
        assert "evoengineer" in algos
        assert "funsearch" in algos

    def test_all_exports(self):
        for name in evotoolkit.__all__:
            assert hasattr(evotoolkit, name), f"Missing export: {name}"
