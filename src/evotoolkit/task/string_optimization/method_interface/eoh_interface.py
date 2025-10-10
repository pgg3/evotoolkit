# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from typing import List
from evotoolkit.core import Solution, EoHInterface
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
        task_description = self.task.get_base_task_description()

        prompt = f"""
{task_description}

1. First, describe your approach and main idea in one sentence. The description must be inside within boxed {{}}.
2. Next, provide your solution as a string following the required format.

Do not give additional explanations.
"""
        return [{'role': 'user', 'content': prompt}]

    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E1 (initialization from population) prompt."""
        task_description = self.task.get_base_task_description()

        # Create prompt content for all individuals
        indivs_prompt = ""
        for i, indi in enumerate(selected_individuals):
            if 'algorithm' in indi.other_info and indi.other_info['algorithm']:
                algorithm_desc = indi.other_info['algorithm']
            else:
                algorithm_desc = f"Solution {i+1}"
            indivs_prompt += f'No. {i + 1} approach and the corresponding solution are:\n{algorithm_desc}\n{indi.sol_string}\n'

        prompt = f"""
{task_description}

I have {len(selected_individuals)} existing solutions as follows:
{indivs_prompt}

Please help me create a new solution that has a totally different form from the given ones.
1. First, describe your new approach in one sentence. The description must be inside within boxed {{}}.
2. Next, provide your solution as a string.

Do not give additional explanations.
"""
        return [{'role': 'user', 'content': prompt}]

    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E2 (guided crossover) prompt."""
        task_description = self.task.get_base_task_description()

        # Create prompt content for all individuals
        indivs_prompt = ""
        for i, indi in enumerate(selected_individuals):
            if 'algorithm' in indi.other_info and indi.other_info['algorithm']:
                algorithm_desc = indi.other_info['algorithm']
            else:
                algorithm_desc = f"Solution {i+1}"
            indivs_prompt += f'No. {i + 1} approach and the corresponding solution are:\n{algorithm_desc}\n{indi.sol_string}\n'

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
        return [{'role': 'user', 'content': prompt}]

    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """Generate M1 (mutation) prompt."""
        task_description = self.task.get_base_task_description()

        if 'algorithm' in individual.other_info and individual.other_info['algorithm']:
            algorithm_desc = individual.other_info['algorithm']
        else:
            algorithm_desc = "Current solution"

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
        return [{'role': 'user', 'content': prompt}]
