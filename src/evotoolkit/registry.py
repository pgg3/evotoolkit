# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Registry system for tasks and algorithms.

This module provides decorators and factory functions to simplify
the creation of evolutionary optimization workflows.
"""

from typing import Dict, Type, Any, Callable
from evotoolkit.core import BaseTask, BaseMethodInterface


# Global registries
_TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}
_ALGORITHM_REGISTRY: Dict[str, Dict[str, Any]] = {}


# Interface naming mapping: algorithm_name -> interface_class_prefix
_INTERFACE_PREFIX_MAP = {
    "eoh": "EoH",
    "evoengineer": "EvoEngineer",
    "funsearch": "FunSearch",
}

# Reverse mapping: interface_class_prefix -> algorithm_name
_INTERFACE_TO_ALGORITHM_MAP = {v: k for k, v in _INTERFACE_PREFIX_MAP.items()}


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

    def decorator(task_class: Type[BaseTask]) -> Type[BaseTask]:
        if name in _TASK_REGISTRY:
            raise ValueError(f"Task '{name}' is already registered")
        _TASK_REGISTRY[name] = task_class
        return task_class

    return decorator


def register_algorithm(name: str, config: Type) -> Callable:
    """
    Decorator to register an algorithm with its config.

    Interface selection is handled automatically based on task type.

    Usage:
        @register_algorithm("eoh", config=EoHConfig)
        class EoH:
            ...

    Args:
        name: Unique name for the algorithm
        config: Config class for this algorithm

    Returns:
        Decorator function
    """

    def decorator(algorithm_class: Type) -> Type:
        if name in _ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm '{name}' is already registered")
        _ALGORITHM_REGISTRY[name] = {"class": algorithm_class, "config": config}
        return algorithm_class

    return decorator


def get_task_class(name: str) -> Type[BaseTask]:
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
        Dictionary with 'class' and 'config' keys

    Raises:
        ValueError: If algorithm name is not registered
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(_ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Algorithm '{name}' not found. Available algorithms: {available}"
        )
    return _ALGORITHM_REGISTRY[name]


def get_interface_class(
    algorithm_name: str, task: BaseTask
) -> Type[BaseMethodInterface]:
    """
    Automatically select and return the appropriate Interface class
    based on algorithm name and task type.

    Args:
        algorithm_name: Name of the algorithm (e.g., 'eoh', 'evoengineer')
        task: Task instance to determine interface type

    Returns:
        Interface class

    Raises:
        ValueError: If interface cannot be found for the algorithm-task combination
    """
    from evotoolkit.task.python_task import PythonTask

    # Determine task type suffix
    if isinstance(task, PythonTask):
        task_suffix = "Python"
        module_path = "evotool.task.python_task"
    else:
        # Check if it's a CUDA task
        try:
            from evotoolkit.task.cuda_engineering import CudaTaskConfig

            if isinstance(task, CudaTaskConfig):
                task_suffix = "Cuda"
                module_path = "evotool.task.cuda_engineering"
            else:
                raise ValueError(f"Unknown task type: {type(task)}")
        except ImportError:
            raise ValueError(f"Unknown task type: {type(task)}")

    # Get interface class name prefix
    if algorithm_name not in _INTERFACE_PREFIX_MAP:
        raise ValueError(
            f"No interface mapping for algorithm '{algorithm_name}'. "
            f"Available: {list(_INTERFACE_PREFIX_MAP.keys())}"
        )

    prefix = _INTERFACE_PREFIX_MAP[algorithm_name]
    interface_class_name = f"{prefix}{task_suffix}Interface"

    # Dynamically import the interface class
    try:
        import importlib

        module = importlib.import_module(module_path)
        interface_class = getattr(module, interface_class_name)
        return interface_class
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Could not find interface '{interface_class_name}' in '{module_path}'. "
            f"Error: {e}"
        )


def infer_algorithm_from_interface(interface: BaseMethodInterface) -> str:
    """
    Infer algorithm name from interface instance.

    Args:
        interface: Interface instance

    Returns:
        Algorithm name (e.g., 'eoh', 'evoengineer', 'funsearch')

    Raises:
        ValueError: If algorithm cannot be inferred from interface
    """
    interface_class_name = interface.__class__.__name__

    # Try to match interface class name patterns
    # e.g., EoHPythonInterface -> EoH, EvoEngineerCudaInterface -> EvoEngineer
    for prefix, algo_name in _INTERFACE_TO_ALGORITHM_MAP.items():
        if interface_class_name.startswith(prefix):
            return algo_name

    raise ValueError(
        f"Cannot infer algorithm from interface '{interface_class_name}'. "
        f"Expected interface name to start with one of: {list(_INTERFACE_TO_ALGORITHM_MAP.keys())}"
    )


def list_tasks() -> list[str]:
    """List all registered task names."""
    return list(_TASK_REGISTRY.keys())


def list_algorithms() -> list[str]:
    """List all registered algorithm names."""
    return list(_ALGORITHM_REGISTRY.keys())
