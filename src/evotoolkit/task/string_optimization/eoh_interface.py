# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import re
from typing import List

from evotoolkit.core import Solution
from evotoolkit.evo_method.eoh import EoHInterface
from evotoolkit.task.string_optimization.string_task import StringTask


class EoHStringInterface(EoHInterface):
    """EoH Adapter for string optimization tasks.

    This class provides EoH (Evolution of Heuristics) algorithm logic for
    string-based tasks like prompt optimization.
    """

    def __init__(self, task: StringTask):
        super().__init__(task)

    def get_prompt_i1(self) -> List[dict]:
        """Generate initialization prompt (I1 operator)."""
        task_description = self.task.spec.prompt

        prompt = f"""
{task_description}

1. First, describe your approach and main idea in one sentence. The description must be inside within boxed {{}}.
2. Next, provide your solution as a string following the required format.

Do not give additional explanations.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E1 (initialization from population) prompt."""
        task_description = self.task.spec.prompt

        # Create prompt content for all individuals
        indivs_prompt = ""
        for i, indi in enumerate(selected_individuals):
            algorithm_desc = indi.metadata.description or f"Solution {i + 1}"
            indivs_prompt += f"No. {i + 1} approach and the corresponding solution are:\n{algorithm_desc}\n{indi.sol_string}\n"

        prompt = f"""
{task_description}

I have {len(selected_individuals)} existing solutions as follows:
{indivs_prompt}

Please help me create a new solution that has a totally different form from the given ones.
1. First, describe your new approach in one sentence. The description must be inside within boxed {{}}.
2. Next, provide your solution as a string.

Do not give additional explanations.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E2 (guided crossover) prompt."""
        task_description = self.task.spec.prompt

        # Create prompt content for all individuals
        indivs_prompt = ""
        for i, indi in enumerate(selected_individuals):
            algorithm_desc = indi.metadata.description or f"Solution {i + 1}"
            indivs_prompt += f"No. {i + 1} approach and the corresponding solution are:\n{algorithm_desc}\n{indi.sol_string}\n"

        prompt = f"""
{task_description}

I have {len(selected_individuals)} existing solutions as follows:
{indivs_prompt}

Please help me create a new solution that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided solutions.
2. Secondly, based on the backbone idea describe your new solution in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, provide your solution as a string.

Do not give additional explanations.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """Generate M1 (mutation) prompt."""
        task_description = self.task.spec.prompt

        algorithm_desc = individual.metadata.description or "Current solution"

        prompt = f"""
{task_description}

The current solution and its approach are as follows:
{algorithm_desc}
{individual.sol_string}

Please assist me in identifying issues with the current solution and make necessary modifications.
1. Firstly, identify and explain the shortcomings of the current solution.
2. Secondly, based on the analysis, describe your new approach in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, provide your improved solution as a string.

Do not give additional explanations.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        """Generate M2 (parameter mutation) prompt."""
        task_description = self.task.spec.prompt

        algorithm_desc = individual.metadata.description or "Current solution"

        prompt = f"""
{task_description}

The current solution and its approach are as follows:
{algorithm_desc}
{individual.sol_string}

Please identify the key configurable parts of the current solution and create a new variant with different settings or wording choices.
1. First, describe your updated approach in one sentence. The description must be inside within boxed {{}}.
2. Next, provide your revised solution as a string.

Do not give additional explanations.
"""
        return [{"role": "user", "content": prompt}]

    def parse_response(self, response_str: str) -> Solution:
        """Parse bracketed rationale and return the remaining string solution."""
        if not response_str or not response_str.strip():
            return self.make_solution("")

        description_match = re.search(r"\{(.*?)\}", response_str, re.DOTALL)
        description = description_match.group(1).strip() if description_match else ""
        sol_string = re.sub(r"\{.*?\}", "", response_str, count=1, flags=re.DOTALL).strip()
        return self.make_solution(sol_string or response_str.strip(), description=description)
