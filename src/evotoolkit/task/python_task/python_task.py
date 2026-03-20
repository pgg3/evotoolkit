# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Python task base class for evolutionary optimization.

This module contains the base class for Python-based tasks.
"""

import traceback
from abc import abstractmethod
from typing import Any

from evotoolkit.core import EvaluationResult, Solution, Task, TaskSpec


class PythonTask(Task):
    """
    Abstract base class for Python-based evolutionary optimization tasks.
    """

    def __init__(self, data, timeout_seconds: float = 30.0):
        """
        Initialize the Python task with input data.

        Args:
            data (Any): Task-specific input data.
            timeout_seconds (float): Execution timeout for code evaluation.
        """
        self.timeout_seconds = timeout_seconds
        super().__init__(data)

    def build_spec(self, data: Any) -> TaskSpec:
        return self.build_python_spec(data)

    def evaluate(self, solution: Solution) -> EvaluationResult:
        """Evaluate Python code contained in a solution."""
        try:
            return self._evaluate_code_impl(solution.sol_string)
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={
                    "error": f"Evaluation error: {str(e)}",
                    "traceback": traceback.format_exc(),
                },
            )

    @abstractmethod
    def build_python_spec(self, data: Any) -> TaskSpec:
        """Build the task specification for this Python task."""

    @abstractmethod
    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """
        Implement specific code evaluation logic.

        Subclasses must implement this method with their specific
        evaluation logic. This method is called by evaluate()
        within a try-catch block.

        Args:
            candidate_code: Python code to evaluate

        Returns:
            EvaluationResult: Result of the evaluation
        """
        pass
