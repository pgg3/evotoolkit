# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for package surfaces intentionally removed in 2.0."""

import importlib

import pytest


@pytest.mark.parametrize(
    ("old_path", "new_path"),
    [
        ("evotoolkit.data", "evotoolkit_tasks.data"),
        (
            "evotoolkit.task.python_task.adversarial_attack",
            "evotoolkit_tasks.python_task.adversarial_attack",
        ),
        (
            "evotoolkit.task.python_task.control_box2d",
            "evotoolkit_tasks.python_task.control_box2d",
        ),
        (
            "evotoolkit.task.python_task.scientific_regression",
            "evotoolkit_tasks.python_task.scientific_regression",
        ),
        (
            "evotoolkit.task.string_optimization.prompt_optimization",
            "evotoolkit_tasks.string_optimization.prompt_optimization",
        ),
        (
            "evotoolkit.evo_method.cann_initer",
            "evotoolkit_tasks.evo_method.cann_initer",
        ),
        ("evotoolkit.task.cuda_engineering", "evotoolkit_tasks.cuda_engineering"),
        ("evotoolkit.task.cann_init", "evotoolkit_tasks.cann_init"),
    ],
)
def test_removed_imports_raise_migration_error(old_path, new_path):
    with pytest.raises(ModuleNotFoundError, match="removed from evotoolkit 2.0.0") as exc_info:
        importlib.import_module(old_path)

    assert new_path in str(exc_info.value)
