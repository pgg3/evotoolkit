# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Python task base class for evolutionary optimization.

This module contains the base class for Python-based tasks, unifying
the functionality of PythonEvaluator and PythonTaskConfig.
"""

import traceback
from abc import abstractmethod

from evotoolkit.core import BaseTask, EvaluationResult


class PythonTask(BaseTask):
    """
    Abstract base class for Python-based evolutionary optimization tasks.

    This class unifies PythonEvaluator and PythonTaskConfig functionality,
    providing a common base for Python code evaluation tasks.
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

    def get_task_type(self) -> str:
        """Get task type as 'Python'."""
        return "Python"

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        """
        Evaluate Python code.

        Default implementation provides basic error handling framework.
        Subclasses should override this method with specific evaluation logic.

        Args:
            candidate_code: Python code to evaluate

        Returns:
            EvaluationResult: Result of the evaluation
        """
        try:
            return self._evaluate_code_impl(candidate_code)
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
    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """
        Implement specific code evaluation logic.

        Subclasses must implement this method with their specific
        evaluation logic. This method is called by evaluate_code
        within a try-catch block.

        Args:
            candidate_code: Python code to evaluate

        Returns:
            EvaluationResult: Result of the evaluation
        """
        pass

    # Abstract methods from BaseTask are still required:
    # - get_base_task_description() -> str
    # - make_init_sol_wo_other_info() -> Solution
