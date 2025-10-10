# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import re
from typing import List

from evotoolkit.core import FunSearchInterface, Solution

from ..cuda_task import CudaTask


class FunSearchCudaInterface(FunSearchInterface):
    def __init__(self, task: CudaTask):
        super().__init__(task)

    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        base_task_description = self.task.get_base_task_description()
        if len(solutions) == 1:
            prompt = f"""
{base_task_description}

Here is the CUDA kernel code example you need to optimize:
```cpp
{solutions[0].sol_string}
```

Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        elif len(solutions) >= 2:
            prompt = f"""
{base_task_description}

Here is a CUDA kernel code example:
```cpp
{solutions[0].sol_string}
```

A better version of the CUDA kernel code example is as follows:
```cpp
{solutions[1].sol_string}
```

Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        else:
            # Fallback if no solutions provided
            prompt = f"""
{base_task_description}

Here is the original CUDA kernel code:
```cpp
{self.task.task_info["cuda_code"]}
```

Propose an optimized CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""

        prompt_content = [{"role": "user", "content": prompt}]
        return prompt_content

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response to extract CUDA code"""
        # Try different code block patterns in order of preference
        patterns = [
            r"```cpp\s*\n(.*?)\n```",  # cpp
            r"```c\+\+\s*\n(.*?)\n```",  # c++
            r"```cuda\s*\n(.*?)\n```",  # cuda
            r"```c\s*\n(.*?)\n```",  # c
            r"```\s*\n(.*?)\n```",  # generic code block
        ]

        # Find all matches using case insensitive search
        for pattern in patterns:
            matches = re.findall(pattern, response_str, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the longest match (likely the most complete implementation)
                return Solution(max(matches, key=len).strip())

        # Last resort: return stripped response
        return Solution(response_str.strip())
