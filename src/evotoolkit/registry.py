# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Registry helpers for core algorithms and extension tasks."""

from typing import Any, Callable, Dict, Type

from evotoolkit.core import Task

# Global registries
_TASK_REGISTRY: Dict[str, Type[Task]] = {}
_ALGORITHM_REGISTRY: Dict[str, Dict[str, Any]] = {}


# Interface naming mapping: algorithm_name -> interface_class_prefix
def register_task(name: str) -> Callable:
    """
    Decorator to register a task class.

    Usage:
        @register_task("FuncApprox")
        class FuncApproxTask(PythonTask):
            ...

    Args:
        name: Unique name for the task

    Returns:
        Decorator function
    """

    def decorator(task_class: Type[Task]) -> Type[Task]:
        if name in _TASK_REGISTRY:
            raise ValueError(f"Task '{name}' is already registered")
        _TASK_REGISTRY[name] = task_class
        return task_class

    return decorator


def register_algorithm(name: str) -> Callable:
    """
    Decorator to register an algorithm.

    Usage:
        @register_algorithm("eoh")
        class EoH:
            ...

    Args:
        name: Unique name for the algorithm
    Returns:
        Decorator function
    """

    def decorator(algorithm_class: Type) -> Type:
        if name in _ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm '{name}' is already registered")
        _ALGORITHM_REGISTRY[name] = {"class": algorithm_class}
        return algorithm_class

    return decorator


def get_task_class(name: str) -> Type[Task]:
    """
    Get a registered task class by name.

    Args:
        name: Task name

    Returns:
        Task class

    Raises:
        ValueError: If task name is not registered
    """
    if name not in _TASK_REGISTRY:
        available = ", ".join(_TASK_REGISTRY.keys())
        raise ValueError(f"Task '{name}' not found. Available tasks: {available}")
    return _TASK_REGISTRY[name]


def get_algorithm_info(name: str) -> Dict[str, Any]:
    """
    Get algorithm class and config by name.

    Args:
        name: Algorithm name

    Returns:
        Dictionary with the registered metadata for the algorithm

    Raises:
        ValueError: If algorithm name is not registered
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(_ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algorithm '{name}' not found. Available algorithms: {available}")
    return _ALGORITHM_REGISTRY[name]


def list_tasks() -> list[str]:
    """List all registered task names."""
    return list(_TASK_REGISTRY.keys())


def list_algorithms() -> list[str]:
    """List all registered algorithm names."""
    return list(_ALGORITHM_REGISTRY.keys())
