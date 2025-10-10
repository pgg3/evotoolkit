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

from evotoolkit.core import BaseTask, EvaluationResult


class StringTask(BaseTask):
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

    def get_task_type(self) -> str:
        """Get task type as 'String'."""
        return "String"

    def evaluate_code(self, candidate_string: str) -> EvaluationResult:
        """
        Evaluate a candidate string solution.

        Note: For compatibility with the framework, we keep the method name
        'evaluate_code', but it actually evaluates strings, not code.

        Args:
            candidate_string: String solution to evaluate

        Returns:
            EvaluationResult: Result of the evaluation
        """
        try:
            return self._evaluate_string_impl(candidate_string)
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
    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        """
        Implement specific string evaluation logic.

        Subclasses must implement this method with their specific
        evaluation logic. This method is called by evaluate_code
        within a try-catch block.

        Args:
            candidate_string: String solution to evaluate

        Returns:
            EvaluationResult: Result of the evaluation
        """
        pass

    # Abstract methods from BaseTask are still required:
    # - get_base_task_description() -> str
    # - make_init_sol_wo_other_info() -> Solution
