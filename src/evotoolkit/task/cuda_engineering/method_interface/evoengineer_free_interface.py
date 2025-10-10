# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import re
from evotoolkit.core import Solution, Operator
from typing import List
from ..cuda_task import CudaTask
from .evoengineer_full_interface import EvoEngineerFullCudaInterface


class EvoEngineerFreeCudaInterface(EvoEngineerFullCudaInterface):
    def __init__(self, task: CudaTask):
        super().__init__(task)
        self.valid_require = 0

    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for CUDA optimization"""
        return [
            Operator("init", 0)
        ]

    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for CUDA optimization"""
        return [
            Operator("init", 0)
        ]

    def get_operator_prompt(self, operator_name: str, selected_individuals: List[Solution], current_best_sol: Solution,
                            random_thoughts: List[str], **kwargs) -> List[dict]:
        """Generate prompt for any operator"""
        task_description = self.task.get_base_task_description()

        if current_best_sol is None:
            current_best_sol = self.make_init_sol()

        if operator_name == "init":
            prompt = f"""# CUDA KERNEL OPTIMIZATION TASK
{task_description}

## BASELINE CODE
```cpp
{current_best_sol.sol_string}
```

## OPTIMIZATION STRATEGY
Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.

## RESPONSE FORMAT:

code:
```cpp
[Your CUDA kernel implementation]
```

## FORMAT REQUIREMENTS:
1. MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
2. The PYBIND11_MODULE inside the code has to be the same as ## BASELINE CODE.
3. The code MUST be wrapped in ```cpp and ``` markers."""
            return [{'role': 'user', 'content': prompt}]
        else:
            raise ValueError(f"Unknown operator: {operator_name}")

    def parse_response(self, response_str: str) -> Solution:
        """Parse response with multiple fallback strategies for free format"""
        if not response_str or not response_str.strip():
            return Solution("", other_info={"name": "raw", "thought": "Failed to parse"})

        content = response_str.strip()

        # Strategy 1: Look for code: block format (expected format)
        code_pattern = r'code:\s*\n*```(?:cpp|c\+\+|cuda)?\s*\n(.*?)```'
        code_match = re.search(code_pattern, content, re.DOTALL | re.IGNORECASE)
        if code_match:
            code = code_match.group(1).strip()
            if code:
                return Solution(code, other_info={"name": "code_block", "thought": "Standard format"})

        # Strategy 2: Look for any cpp/cuda code block
        code = self._extract_any_code_block(content)
        if code:
            return Solution(code, other_info={"name": "extracted", "thought": "Code block fallback"})

        # Strategy 3: Raw content (last resort)
        return Solution(content, other_info={"name": "raw", "thought": "Failed to parse"})