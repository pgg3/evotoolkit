# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""FunSearch Interface for CANN Init Task"""

import re
from typing import List

from evotoolkit.core import FunSearchInterface, Solution

from ..cann_init_task import CANNInitTask


class FunSearchCANNInterface(FunSearchInterface):
    """FunSearch interface for CANN kernel optimization"""

    def __init__(self, task: CANNInitTask):
        super().__init__(task)

    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate FunSearch prompt based on available solutions"""
        base_task_description = self.task.get_base_task_description()

        if len(solutions) == 1:
            prompt = f"""
{base_task_description}

Here is an Ascend C kernel code example you need to optimize:
```cpp
{solutions[0].sol_string}
```

Propose a new Ascend C kernel code which aims to improve the performance (reduce runtime) of the operation, while ensuring the kernel returns the correct result.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

MAKE SURE THE PROPOSAL CODE IS VALID ASCEND C CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        elif len(solutions) >= 2:
            prompt = f"""
{base_task_description}

Here is an Ascend C kernel code example:
```cpp
{solutions[0].sol_string}
```

A better version of the Ascend C kernel code is as follows:
```cpp
{solutions[1].sol_string}
```

Propose a new Ascend C kernel code which aims to further improve the performance (reduce runtime) of the operation, while ensuring the kernel returns the correct result.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

MAKE SURE THE PROPOSAL CODE IS VALID ASCEND C CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""
        else:
            # Fallback if no solutions provided
            prompt = f"""
{base_task_description}

Propose an optimized Ascend C kernel code which aims to improve the performance (reduce runtime) of the operation, while ensuring the kernel returns the correct result.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

MAKE SURE THE PROPOSAL CODE IS VALID ASCEND C CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""

        prompt_content = [{"role": "user", "content": prompt}]
        return prompt_content

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response to extract Ascend C kernel code"""
        # Try different code block patterns in order of preference
        patterns = [
            r"```cpp\s*\n(.*?)\n```",  # cpp
            r"```c\+\+\s*\n(.*?)\n```",  # c++
            r"```c\s*\n(.*?)\n```",  # c
            r"```ascend\s*\n(.*?)\n```",  # ascend
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
