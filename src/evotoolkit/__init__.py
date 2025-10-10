# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
EvoTool - Evolutionary Optimization Toolkit

A Python library for LLM-driven code evolutionary optimization.
"""

from typing import Any
from evotoolkit.registry import (
    get_algorithm_info,
    infer_algorithm_from_interface,
    list_tasks,
    list_algorithms,
)
from evotoolkit.core import BaseMethodInterface

# Import tasks and algorithms to trigger registration decorators
from evotoolkit.evo_method.eoh import EoH
from evotoolkit.evo_method.evoengineer import EvoEngineer
from evotoolkit.evo_method.funsearch import FunSearch

__author__ = "Ping Guo"

# Read version from package metadata (set in pyproject.toml)
try:
    from importlib.metadata import version

    __version__ = version("evotoolkit")
except Exception:
    # Fallback for development/editable install
    __version__ = "0.3.0b1.dev"


def solve(
    interface: BaseMethodInterface, output_path: str = "./results", **kwargs
) -> Any:
    """
    Factory method to create and run an evolutionary optimization workflow.

    This is the main entry point for using evotool with an explicit, unambiguous API.
    Users must explicitly create task and interface instances before calling solve().

    Args:
        interface (BaseMethodInterface): Interface instance (e.g.,
            `EoHPythonInterface`, `EvoEngineerPythonInterface`). Must be
            explicitly created with a task instance. The algorithm is
            automatically inferred from the interface type.
        output_path (str): Path to save results.
        **kwargs (Any): Additional parameters passed to algorithm config
            (e.g., `max_generations`, `max_sample_nums`, `pop_size`,
            `running_llm`).

    Returns:
        Any: Best solution found during the evolutionary optimization run.

    Example:
        # Create task instance explicitly
        task = FuncApproxTask(x_data, y_noisy, y_true)

        # Create interface instance explicitly
        interface = EoHPythonInterface(task)

        # Call solve with explicit interface
        result = evotool.solve(
            interface=interface,
            output_path='./results',
            running_llm=llm_api,
            max_generations=5,
            max_sample_nums=10
        )
    """
    # Step 1: Infer algorithm from interface
    algorithm_name = infer_algorithm_from_interface(interface)

    # Step 2: Get algorithm info from registry
    algo_info = get_algorithm_info(algorithm_name)
    algorithm_class = algo_info["class"]
    config_class = algo_info["config"]

    # Step 3: Create config with all parameters
    # Note: task is accessed via interface.task
    config = config_class(interface=interface, output_path=output_path, **kwargs)

    # Step 4: Create and run algorithm
    algorithm_instance = algorithm_class(config)
    algorithm_instance.run()

    # Step 5: Get the best solution from the run
    best_solution = algorithm_instance._get_best_sol(
        algorithm_instance.run_state_dict.sol_history
    )

    return best_solution


# Export public API
__all__ = ["solve", "list_tasks", "list_algorithms", "__version__", "__author__"]
