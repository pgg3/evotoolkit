# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Base task classes for evolutionary optimization.

This module contains abstract base classes that unify task evaluation
and configuration into a single concept, simplifying the API while
maintaining all existing abstractions.
"""

from abc import ABC, abstractmethod

from .solution import EvaluationResult, Solution


class BaseTask(ABC):
    """
    Abstract base class for evolutionary optimization tasks.

    This class unifies the functionality of BaseEvaluator and BaseTaskConfig
    into a single concept, providing both evaluation capabilities and task
    configuration in one place.
    """

    def __init__(self, data):
        """
        Initialize the task with input data.

        Args:
            data (Any): Task-specific input data (format varies by task type).
        """
        self._process_data(data)

    def _process_data(self, data):
        """
        Process input data and set up task_info.

        This method should be overridden by subclasses to handle
        task-specific data processing and create the task_info dict.

        Args:
            data (Any): Task-specific input data.
        """
        self.data = data
        self.task_info = {}  # Subclasses should populate this

    # === Abstract methods from BaseEvaluator ===

    @abstractmethod
    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        """
        Evaluate a candidate code solution and return evaluation result.

        Args:
            candidate_code: The code to evaluate

        Returns:
            EvaluationResult: Result of the evaluation
        """
        pass

    # === Abstract methods from BaseTaskConfig ===

    @abstractmethod
    def get_base_task_description(self) -> str:
        """
        Get the base task description for prompt generation.

        Returns:
            str: Task description text
        """
        pass

    @abstractmethod
    def make_init_sol_wo_other_info(self) -> Solution:
        """
        Create initial solution from task info without other_info.

        Returns:
            Solution: Initial solution for this task
        """
        pass

    # === Optional methods that subclasses can override ===

    def get_task_type(self) -> str:
        """
        Get the type of this task (e.g., 'Python', 'Cuda').

        Default implementation returns 'Python'. Subclasses should
        override if they represent different task types.

        Returns:
            str: Task type identifier
        """
        return "Python"

    def get_task_info(self) -> dict:
        """
        Get the task_info dictionary.

        Returns:
            dict: Task information dictionary
        """
        return self.task_info
