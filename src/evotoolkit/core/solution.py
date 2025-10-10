# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Core data structures for EvoTool.

This module contains the fundamental data structures used throughout the framework:
- Solution: Represents a candidate solution with evaluation results
- EvaluationResult: Stores evaluation outcome and metrics
"""


class EvaluationResult:
    """Stores the result of evaluating a solution."""

    def __init__(self, valid, score, additional_info):
        self.valid = valid
        self.score = score
        self.additional_info = additional_info


class Solution:
    """Represents a candidate solution in the evolutionary process."""

    def __init__(self, sol_string, other_info: dict = None, evaluation_res: EvaluationResult = None):
        self.sol_string = sol_string
        self.other_info = other_info
        self.evaluation_res = evaluation_res