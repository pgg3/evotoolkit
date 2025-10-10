# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from typing import List

from evotoolkit.core import FunSearchInterface, Solution
from evotoolkit.task.string_optimization.string_task import StringTask


class FunSearchStringInterface(FunSearchInterface):
    """FunSearch Adapter for string optimization tasks.

    This class provides FunSearch algorithm logic for string-based tasks
    like prompt optimization.
    """

    def __init__(self, task: StringTask):
        super().__init__(task)

    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate prompt based on multiple solutions"""
        task_description = self.task.get_base_task_description()

        if len(solutions) == 1:
            prompt = f"""
{task_description}

Here is an example solution you need to improve:
{solutions[0].sol_string}

Propose a new solution which performs better than the above solution.
Provide only the solution string following the required format.

MAKE SURE THE PROPOSED SOLUTION FOLLOWS THE REQUIRED FORMAT.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        elif len(solutions) >= 2:
            prompt = f"""
{task_description}

Here is a solution:
{solutions[0].sol_string}

A better version of the solution is as follows:
{solutions[1].sol_string}

Propose a new solution which performs better than the above solutions.
Provide only the solution string following the required format.

MAKE SURE THE PROPOSED SOLUTION FOLLOWS THE REQUIRED FORMAT.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        else:
            # Fallback if no solutions provided
            prompt = f"""
{task_description}

Propose a new solution following the required format.

MAKE SURE THE PROPOSED SOLUTION FOLLOWS THE REQUIRED FORMAT.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""

        return [{"role": "user", "content": prompt}]
