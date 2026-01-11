# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
EoH (Evolution of Heuristics) interface for control policy evolution.

This interface provides 5 evolutionary operators:
- I1: Initialize a new policy from scratch
- E1: Create a completely different policy (exploration)
- E2: Combine policies based on common ideas (guided crossover)
- M1: Modify existing policy logic (mutation)
- M2: Adjust policy parameters (parameter mutation)
"""

import re
from typing import List

from evotoolkit.core import EoHInterface, Solution

# Control-specific optimization insights
CONTROL_INSIGHTS = [
    "Use PD control: action = Kp * error + Kd * derivative for smooth control",
    "Side engines are cheap (-0.03/frame), main engine is expensive (-0.3/frame)",
    "Prioritize not crashing over fuel efficiency - survival first",
    "The angle and angular velocity form a coupled system - control them together",
    "Consider state derivatives (acceleration) for predictive control",
    "Add dead-zones around target values to prevent oscillation",
    "Height and vertical velocity determine the urgency of main engine thrust",
    "Final approach requires: x ≈ 0, angle ≈ 0, vx ≈ 0, vy small negative",
    "Both legs must touch ground for maximum landing bonus (+20 points)",
    "State-dependent thresholds can adapt control to different flight phases",
]


class EoHControlInterface(EoHInterface):
    """
    EoH interface specialized for control policy evolution.

    Provides domain-specific prompts for evolving interpretable
    control policies in environments like LunarLander.
    """

    def __init__(self, task):
        """
        Initialize EoH interface for control tasks.

        Args:
            task: A control task (e.g., LunarLanderTask)
        """
        super().__init__(task)

    def _format_policy_info(self, solution: Solution, index: int = None) -> str:
        """Format a solution's information for prompts."""
        # Get algorithm description
        if "algorithm" in solution.other_info and solution.other_info["algorithm"]:
            algorithm_desc = solution.other_info["algorithm"]
        else:
            algorithm_desc = "No description available"

        # Get performance metrics
        score = 0.0
        success_rate = "N/A"
        if solution.evaluation_res:
            score = solution.evaluation_res.score
            if solution.evaluation_res.additional_info:
                sr = solution.evaluation_res.additional_info.get("success_rate")
                if sr is not None:
                    success_rate = f"{sr:.1%}"

        prefix = f"No. {index}" if index else "Policy"
        return f"""{prefix}:
- Strategy: {algorithm_desc}
- Average Reward: {score:.2f}
- Success Rate: {success_rate}
```python
{solution.sol_string}
```"""

    def get_prompt_i1(self) -> List[dict]:
        """
        Generate initialization prompt (I1 operator).

        Creates a new policy from scratch based on task description.
        """
        task_description = self.task.get_base_task_description()

        prompt = f"""## CONTROL POLICY EVOLUTION TASK

{task_description}

## YOUR TASK
Design a control policy function that achieves high reward in this environment.

## RESPONSE FORMAT
1. First, describe your control strategy in one sentence. The description must be inside curly braces {{}}.
2. Then, implement the policy function:
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your implementation
    return action
```

## TIPS
- Analyze the state space to understand what each dimension means
- Consider different control strategies: rule-based, PD control, state machines
- Balance between landing success and fuel efficiency

Do not provide additional explanations beyond the required format.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """
        Generate exploration crossover prompt (E1 operator).

        Creates a completely different policy from existing ones.
        """
        task_description = self.task.get_base_task_description()

        # Format existing policies
        policies_info = ""
        for i, indi in enumerate(selected_individuals, 1):
            policies_info += self._format_policy_info(indi, i) + "\n\n"

        prompt = f"""## CONTROL POLICY EVOLUTION TASK

{task_description}

## EXISTING POLICIES
I have {len(selected_individuals)} existing control policies:

{policies_info}

## YOUR TASK
Create a NEW control policy that uses a COMPLETELY DIFFERENT approach from all the existing ones.
- Do NOT just modify parameters or thresholds
- Design a fundamentally different control strategy
- Think about alternative ways to interpret the state and select actions

## RESPONSE FORMAT
1. First, describe your new control strategy in one sentence. The description must be inside curly braces {{}}.
2. Then, implement the policy function:
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your completely different implementation
    return action
```

Do not provide additional explanations beyond the required format.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """
        Generate guided crossover prompt (E2 operator).

        Combines policies based on common backbone ideas.
        """
        task_description = self.task.get_base_task_description()

        # Format existing policies
        policies_info = ""
        for i, indi in enumerate(selected_individuals, 1):
            policies_info += self._format_policy_info(indi, i) + "\n\n"

        prompt = f"""## CONTROL POLICY EVOLUTION TASK

{task_description}

## EXISTING POLICIES
I have {len(selected_individuals)} existing control policies:

{policies_info}

## YOUR TASK
Create a new policy that COMBINES the best ideas from the existing policies:

1. First, identify the COMMON BACKBONE IDEAS shared by these policies
   - What control principles do they use?
   - What state variables do they prioritize?
   - What action selection logic works well?

2. Then, create a HYBRID policy that:
   - Merges successful patterns from multiple policies
   - Addresses weaknesses found in individual policies
   - Creates a more robust combined strategy

## RESPONSE FORMAT
1. First, describe your hybrid control strategy in one sentence. The description must be inside curly braces {{}}.
2. Then, implement the policy function:
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your hybrid implementation combining best ideas
    return action
```

Do not provide additional explanations beyond the required format.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """
        Generate mutation prompt (M1 operator).

        Modifies existing policy logic to explore nearby solutions.
        """
        task_description = self.task.get_base_task_description()
        policy_info = self._format_policy_info(individual)

        # Select a random insight for guidance
        import random
        insight = random.choice(CONTROL_INSIGHTS)

        prompt = f"""## CONTROL POLICY EVOLUTION TASK

{task_description}

## CURRENT POLICY
{policy_info}

## OPTIMIZATION INSIGHT
{insight}

## YOUR TASK
Create a MODIFIED VERSION of the current policy:
- Change the control logic structure (not just parameters)
- Add, remove, or reorganize decision conditions
- Improve edge case handling
- Address potential weaknesses in the current approach

The new policy should be DIFFERENT but RELATED to the original.

## RESPONSE FORMAT
1. First, describe your modified control strategy in one sentence. The description must be inside curly braces {{}}.
2. Then, implement the policy function:
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your modified implementation
    return action
```

Do not provide additional explanations beyond the required format.
"""
        return [{"role": "user", "content": prompt}]

    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        """
        Generate parameter mutation prompt (M2 operator).

        Adjusts policy parameters while keeping structure similar.
        """
        task_description = self.task.get_base_task_description()
        policy_info = self._format_policy_info(individual)

        prompt = f"""## CONTROL POLICY EVOLUTION TASK

{task_description}

## CURRENT POLICY
{policy_info}

## YOUR TASK
Create a new policy by TUNING THE PARAMETERS of the current policy:
- Identify the main control parameters (thresholds, gains, coefficients)
- Adjust these values to potentially improve performance
- Keep the overall structure and logic similar

Examples of parameters to tune:
- Velocity thresholds (e.g., vy < -0.5 → vy < -0.3)
- Position thresholds (e.g., x > 0.2 → x > 0.15)
- Control gains (e.g., angle * 0.5 → angle * 0.7)
- Priority orderings

## RESPONSE FORMAT
1. First, describe your parameter changes in one sentence. The description must be inside curly braces {{}}.
2. Then, implement the policy function:
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your parameter-tuned implementation
    return action
```

Do not provide additional explanations beyond the required format.
"""
        return [{"role": "user", "content": prompt}]

    def parse_response(self, response_str: str) -> Solution:
        """
        Parse LLM response to extract policy code and strategy description.

        Returns:
            Solution with code and algorithm description in other_info
        """
        # Extract strategy description from {braces}
        try:
            pattern = r"\{([^{}]*)\}"
            bracketed_texts = re.findall(pattern, response_str, re.DOTALL)
            algorithm = bracketed_texts[0].strip() if bracketed_texts else None
        except Exception:
            algorithm = None

        # Remove algorithm part before extracting code
        response_without_algorithm = response_str
        if algorithm:
            response_without_algorithm = response_str.replace(
                "{" + algorithm + "}", "", 1
            )

        # Extract Python code block
        patterns = [
            r"```python\s*\n(.*?)\n```",
            r"```Python\s*\n(.*?)\n```",
            r"```py\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
        ]

        code = ""
        for pattern in patterns:
            matches = re.findall(
                pattern, response_without_algorithm, re.DOTALL | re.IGNORECASE
            )
            if matches:
                # Return the longest match (most complete implementation)
                code = max(matches, key=len).strip()
                break

        if not code:
            # Last resort: return stripped response
            code = response_without_algorithm.strip()

        return Solution(code, other_info={"algorithm": algorithm})
