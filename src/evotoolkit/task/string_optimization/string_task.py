# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
String task base class for evolutionary optimization.

This module contains the base class for string-based tasks, where solutions
are represented as strings (e.g., prompts, templates, configurations).
"""

import traceback
from abc import abstractmethod
from typing import Any

from evotoolkit.core import EvaluationResult, Solution, Task, TaskSpec


class StringTask(Task):
    """
    Abstract base class for string-based evolutionary optimization tasks.

    Unlike PythonTask or CudaTask which evaluate code, StringTask directly
    evaluates string solutions (e.g., prompts, templates, configurations).
    """

    def __init__(self, data: Any, timeout_seconds: float = 30.0):
        """
        Initialize the string task with input data.

        Args:
            data: Task-specific input data
            timeout_seconds: Execution timeout for evaluation
        """
        self.timeout_seconds = timeout_seconds
        super().__init__(data)

    def build_spec(self, data: Any) -> TaskSpec:
        return self.build_string_spec(data)

    def evaluate(self, solution: Solution) -> EvaluationResult:
        """Evaluate a candidate string solution."""
        try:
            return self._evaluate_string_impl(solution.sol_string)
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
    def build_string_spec(self, data: Any) -> TaskSpec:
        """Build the task specification for this string task."""

    @abstractmethod
    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        """
        Implement specific string evaluation logic.

        Subclasses must implement this method with their specific
        evaluation logic. This method is called by evaluate()
        within a try-catch block.

        Args:
            candidate_string: String solution to evaluate

        Returns:
            EvaluationResult: Result of the evaluation
        """
        pass
