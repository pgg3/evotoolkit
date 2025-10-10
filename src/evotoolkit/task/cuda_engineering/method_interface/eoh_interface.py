# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import re
from typing import List

from evotoolkit.core import EoHInterface, Solution

from ..cuda_task import CudaTask


class EoHCudaInterface(EoHInterface):
    def __init__(self, task: CudaTask):
        super().__init__(task)

    def _get_base_task_description(self) -> str:
        base_task_description = self.task.get_base_task_description()
        baseline_code = self.task.task_info.get("cuda_code", "")

        return f"""
{base_task_description}

Here is the CUDA kernel code example you need to optimize:
```cpp
{baseline_code}
```
"""

    def get_prompt_i1(self) -> List[dict]:
        """Generate initialization prompt (I1 operator)"""
        task_description = self._get_base_task_description()

        prompt = f"""
{task_description}

1. First, describe your new implementation and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, give the optimized kernel implementation:
```cpp
[Your kernel implementation]
```
Do not give additional explanations.

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E1 (crossover) prompt"""
        task_description = self._get_base_task_description()

        # Create prompt content for all individuals
        indivs_prompt = ""
        for i, indi in enumerate(selected_individuals):
            if "algorithm" in indi.other_info and indi.other_info["algorithm"]:
                algorithm_desc = indi.other_info["algorithm"]
            else:
                algorithm_desc = f"Kernel implementation {i + 1}"
            indivs_prompt += f"No. {i + 1} kernel implementation and the corresponding code are:\n{algorithm_desc}\n{indi.sol_string}\n"

        prompt = f"""
{task_description}

I have {len(selected_individuals)} existing kernel implementations with their codes as follows:
{indivs_prompt}

Please help me create a new kernel implementation that has a totally different form from the given ones.
1. First, describe your new kernel implementation and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the kernel:
```cpp
[Your kernel implementation]
```
Do not give additional explanations.

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E2 (guided crossover) prompt"""
        task_description = self._get_base_task_description()

        # Create prompt content for all individuals
        indivs_prompt = ""
        for i, indi in enumerate(selected_individuals):
            if "algorithm" in indi.other_info and indi.other_info["algorithm"]:
                algorithm_desc = indi.other_info["algorithm"]
            else:
                algorithm_desc = f"Kernel implementation {i + 1}"
            indivs_prompt += f"No. {i + 1} kernel implementation and the corresponding code are:\n{algorithm_desc}\n{indi.sol_string}\n"

        prompt = f"""
{task_description}

I have {len(selected_individuals)} existing kernel implementations with their codes as follows:
{indivs_prompt}

Please help me create a new kernel implementation that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided kernel implementations.
2. Secondly, based on the backbone idea describe your new kernel implementation in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the kernel:
```cpp
[Your kernel implementation]
```
Do not give additional explanations.

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """Generate M1 (mutation) prompt"""
        task_description = self._get_base_task_description()

        if "algorithm" in individual.other_info and individual.other_info["algorithm"]:
            algorithm_desc = individual.other_info["algorithm"]
        else:
            algorithm_desc = "Current kernel implementation"

        prompt = f"""
{task_description}

I have one kernel implementation with its code as follows. Kernel implementation description:
{algorithm_desc}
Code:
{individual.sol_string}

Please assist me in creating a new kernel implementation that has a different form but can be a modified version of the kernel implementation provided.
1. First, describe your new kernel implementation and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the kernel:
```cpp
[Your kernel implementation]
```
Do not give additional explanations.

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        """Generate M2 (parameter mutation) prompt"""
        task_description = self._get_base_task_description()

        if "algorithm" in individual.other_info and individual.other_info["algorithm"]:
            algorithm_desc = individual.other_info["algorithm"]
        else:
            algorithm_desc = "Current kernel implementation"

        prompt = f"""
{task_description}

I have one kernel implementation with its code as follows. Kernel implementation description:
{algorithm_desc}
Code:
{individual.sol_string}

Please identify the main kernel implementation parameters and assist me in creating a new kernel implementation that has a different parameter settings of the kernel implementation provided.
1. First, describe your new kernel implementation and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the kernel:
```cpp
[Your kernel implementation]
```
Do not give additional explanations.

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        return [{"role": "user", "content": prompt}]

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response to extract solution string and algorithm description"""
        # Extract algorithm/thought from response using pattern matching
        try:
            pattern = r"\{.*?\}"
            bracketed_texts = re.findall(pattern, response_str, re.DOTALL)
            algorithm = bracketed_texts[0] if bracketed_texts else None
        except Exception:
            algorithm = None

        # Remove only the algorithm part from response before code extraction
        response_without_algorithm = response_str
        if algorithm:
            # Remove only the specific algorithm part from the response
            response_without_algorithm = response_str.replace(algorithm, "", 1)

        # Try different code block patterns in order of preference
        patterns = [
            r"```cpp\s*\n(.*?)\n```",  # cpp
            r"```c\+\+\s*\n(.*?)\n```",  # c++
            r"```cuda\s*\n(.*?)\n```",  # cuda
            r"```c\s*\n(.*?)\n```",  # c
            r"```\s*\n(.*?)\n```",  # generic code block
        ]

        # Find all matches using case insensitive search
        code = ""
        for pattern in patterns:
            matches = re.findall(
                pattern, response_without_algorithm, re.DOTALL | re.IGNORECASE
            )
            if matches:
                # Return the longest match (likely the most complete implementation)
                code = max(matches, key=len).strip()
                break

        if not code:
            # Last resort: return stripped response without algorithm
            code = response_without_algorithm.strip()

        # Store algorithm description in the solution (this would need to be handled elsewhere)
        # For now, we just return the code
        other_info = {"algorithm": algorithm}
        return Solution(code, other_info=other_info)
