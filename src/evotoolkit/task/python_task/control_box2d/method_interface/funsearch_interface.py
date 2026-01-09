# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
FunSearch interface for control policy evolution.

FunSearch uses a simple single-prompt approach that shows the LLM
existing solutions and asks for improvements. It's straightforward
and effective for incremental optimization.
"""

import re
from typing import List

from evotoolkit.core import FunSearchInterface, Solution


class FunSearchControlInterface(FunSearchInterface):
    """
    FunSearch interface specialized for control policy evolution.

    Uses a simple prompt structure that shows existing policies
    and asks the LLM to propose better ones.
    """

    def __init__(self, task):
        """
        Initialize FunSearch interface for control tasks.

        Args:
            task: A control task (e.g., LunarLanderTask)
        """
        super().__init__(task)

    def _get_performance_summary(self, solution: Solution) -> str:
        """Get a brief performance summary for a solution."""
        if not solution.evaluation_res:
            return "Score: unknown"

        score = solution.evaluation_res.score
        info = solution.evaluation_res.additional_info or {}

        success_rate = info.get("success_rate")
        if success_rate is not None:
            return f"Average Reward: {score:.2f}, Success Rate: {success_rate:.1%}"
        return f"Average Reward: {score:.2f}"

    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """
        Generate prompt based on existing solutions.

        Args:
            solutions: List of existing solutions to improve upon

        Returns:
            List of message dicts for LLM API
        """
        task_description = self.task.get_base_task_description()

        if len(solutions) == 1:
            # Single solution: ask for improvement
            perf = self._get_performance_summary(solutions[0])

            prompt = f"""## CONTROL POLICY OPTIMIZATION

{task_description}

## CURRENT POLICY
{perf}

```python
{solutions[0].sol_string}
```

## YOUR TASK
Propose a NEW control policy that achieves HIGHER reward than the current one.

Consider:
- Better interpretation of state variables
- Smarter action selection logic
- Improved handling of edge cases
- Better balance between landing success and fuel efficiency

## RESPONSE FORMAT
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your improved implementation
    return action
```

IMPORTANT: Only output valid Python code in the code block. No explanations needed.
"""

        elif len(solutions) >= 2:
            # Multiple solutions: show progression
            worse_sol = solutions[0]
            better_sol = solutions[1]

            worse_perf = self._get_performance_summary(worse_sol)
            better_perf = self._get_performance_summary(better_sol)

            prompt = f"""## CONTROL POLICY OPTIMIZATION

{task_description}

## POLICY PROGRESSION

### Previous Policy (Worse)
{worse_perf}

```python
{worse_sol.sol_string}
```

### Current Policy (Better)
{better_perf}

```python
{better_sol.sol_string}
```

## YOUR TASK
Analyze the improvement from the worse to the better policy, then propose an EVEN BETTER policy.

Think about:
- What made the better policy improve?
- What weaknesses remain?
- How can you push the performance even higher?

## RESPONSE FORMAT
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your even better implementation
    return action
```

IMPORTANT: Only output valid Python code in the code block. No explanations needed.
"""

        else:
            # Fallback: no solutions provided
            prompt = f"""## CONTROL POLICY OPTIMIZATION

{task_description}

## YOUR TASK
Design an effective control policy function for this task.

## RESPONSE FORMAT
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your implementation
    return action
```

IMPORTANT: Only output valid Python code in the code block. No explanations needed.
"""

        return [{"role": "user", "content": prompt}]

    def parse_response(self, response_str: str) -> Solution:
        """
        Parse LLM response to extract policy code.

        Returns:
            Solution containing the extracted code
        """
        # Try different code block patterns
        patterns = [
            r"```python\s*\n(.*?)\n```",
            r"```Python\s*\n(.*?)\n```",
            r"```py\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_str, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the longest match (most complete implementation)
                return Solution(max(matches, key=len).strip())

        # Last resort: return stripped response
        return Solution(response_str.strip())
