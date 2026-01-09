# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
EvoEngineer interface for control policy evolution.

EvoEngineer provides structured prompts with rich metadata:
- Performance analysis (episode statistics)
- Optimization insights (domain knowledge)
- Structured response format (name, code, thought)

Operators:
- init: Initialize from baseline with insights
- crossover: Combine two parent policies
- mutation: Modify a single policy
"""

import random
import re
from typing import List

from evotoolkit.core import EvoEngineerInterface, Operator, Solution

# Domain-specific optimization insights for control tasks
CONTROL_INSIGHTS = [
    # State interpretation
    "The state is [x, y, vx, vy, angle, angular_vel, left_leg, right_leg] - use all dimensions wisely",
    "Vertical velocity (vy) being negative means falling - more negative = falling faster",
    "The angle is in radians: 0 = upright, positive = tilted right, negative = tilted left",
    "Leg contacts (state[6], state[7]) indicate if the lander has touched ground",

    # Control theory
    "PD control: action proportional to error + derivative can provide smooth control",
    "Consider using state derivatives (change in velocity) for predictive control",
    "The angle and angular velocity are coupled - control them together for stability",
    "Dead-zones around target values (e.g., |angle| < 0.05 = do nothing) prevent oscillation",

    # Action selection
    "Action 1 (left engine) pushes RIGHT, Action 3 (right engine) pushes LEFT - counterintuitive!",
    "Main engine (action 2) is expensive (-0.3/frame) but essential for slowing descent",
    "Side engines (actions 1,3) are cheap (-0.03/frame) - use them freely for fine control",
    "Action 0 (do nothing) saves fuel - use it when already on the right trajectory",

    # Strategy tips
    "Prioritize survival (not crashing) over fuel efficiency in early evolution",
    "Height (y) determines urgency: closer to ground = more aggressive control needed",
    "Final approach: aim for x ≈ 0, vx ≈ 0, angle ≈ 0, vy slightly negative",
    "Both legs touching ground gives +20 bonus - land flat, not tilted",
    "State-dependent thresholds adapt control to different flight phases",

    # Common mistakes
    "Firing main engine when tilted wastes fuel and can push sideways",
    "Over-correcting angle causes oscillation - use angular velocity as damping",
    "Ignoring horizontal velocity (vx) leads to drifting past the landing pad",
    "Starting main engine too late causes hard landings or crashes",
]


def generate_episode_analysis(eval_result) -> str:
    """Generate detailed episode analysis from evaluation result."""
    if not eval_result or not eval_result.additional_info:
        return "No episode data available."

    info = eval_result.additional_info
    score = eval_result.score

    # Basic performance metrics
    analysis = f"""### Performance Summary
- **Average Reward:** {score:.2f}
- **Std Deviation:** {info.get('std_reward', 'N/A')}
- **Min / Max Reward:** {info.get('min_reward', 'N/A')} / {info.get('max_reward', 'N/A')}
- **Success Rate:** {info.get('success_rate', 0):.1%}
- **Average Episode Length:** {info.get('avg_length', 'N/A')} steps"""

    # Add episode details if available
    all_rewards = info.get('all_rewards', [])
    if all_rewards:
        successes = sum(1 for r in all_rewards if r > 100)
        crashes = sum(1 for r in all_rewards if r < -50)
        analysis += f"""

### Episode Breakdown
- Successful landings (reward > 100): {successes}/{len(all_rewards)}
- Likely crashes (reward < -50): {crashes}/{len(all_rewards)}
- Episode rewards: {[f'{r:.0f}' for r in all_rewards[:5]]}{'...' if len(all_rewards) > 5 else ''}"""

    return analysis


def select_insights(eval_result, n: int = 3) -> List[str]:
    """Select relevant insights based on current performance."""
    insights = []

    if eval_result and eval_result.additional_info:
        info = eval_result.additional_info
        score = eval_result.score
        success_rate = info.get('success_rate', 0)

        # Add insights based on performance
        if success_rate < 0.3:
            insights.append("Prioritize survival (not crashing) over fuel efficiency in early evolution")
        if score < 0:
            insights.append("Focus on basic control: slow descent, stay upright, aim for center")
        if success_rate > 0.5 and score < 200:
            insights.append("Both legs touching ground gives +20 bonus - land flat, not tilted")

    # Fill remaining slots with random insights
    remaining = n - len(insights)
    if remaining > 0:
        available = [i for i in CONTROL_INSIGHTS if i not in insights]
        insights.extend(random.sample(available, min(remaining, len(available))))

    return insights[:n]


class EvoEngineerControlInterface(EvoEngineerInterface):
    """
    EvoEngineer interface specialized for control policy evolution.

    Provides rich context including:
    - Detailed episode analysis
    - Domain-specific optimization insights
    - Structured name/code/thought response format
    """

    def __init__(self, task):
        """
        Initialize EvoEngineer interface for control tasks.

        Args:
            task: A control task (e.g., LunarLanderTask)
        """
        super().__init__(task)

    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators."""
        return [Operator("init", 0)]

    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring generation operators."""
        return [
            Operator("crossover", 2),
            Operator("mutation", 1),
        ]

    def get_operator_prompt(
        self,
        operator_name: str,
        selected_individuals: List[Solution],
        current_best_sol: Solution,
        random_thoughts: List[str],
        **kwargs,
    ) -> List[dict]:
        """
        Generate operator-specific prompt with rich context.

        Args:
            operator_name: One of "init", "crossover", "mutation"
            selected_individuals: Parent solutions for crossover/mutation
            current_best_sol: Current best solution in population
            random_thoughts: Additional optimization hints

        Returns:
            List of message dicts for LLM API
        """
        task_description = self.task.get_base_task_description()

        # Ensure we have a baseline solution
        if current_best_sol is None:
            current_best_sol = self.make_init_sol()

        # Get episode analysis
        episode_analysis = generate_episode_analysis(current_best_sol.evaluation_res)

        # Get or generate insights
        if random_thoughts and len(random_thoughts) > 0:
            insights = random_thoughts
        else:
            insights = select_insights(current_best_sol.evaluation_res, n=3)

        insights_text = "\n".join([f"- {thought}" for thought in insights])

        if operator_name == "init":
            return self._get_init_prompt(
                task_description, current_best_sol, episode_analysis, insights_text
            )
        elif operator_name == "crossover":
            return self._get_crossover_prompt(
                task_description, current_best_sol, selected_individuals,
                episode_analysis, insights_text
            )
        elif operator_name == "mutation":
            return self._get_mutation_prompt(
                task_description, current_best_sol, selected_individuals[0],
                episode_analysis, insights_text
            )
        else:
            raise ValueError(f"Unknown operator: {operator_name}")

    def _get_init_prompt(
        self,
        task_description: str,
        current_best_sol: Solution,
        episode_analysis: str,
        insights_text: str,
    ) -> List[dict]:
        """Generate initialization prompt."""
        prompt = f"""# CONTROL POLICY OPTIMIZATION TASK

{task_description}

## BASELINE POLICY
**Name:** {current_best_sol.other_info.get("name", "baseline")}
**Average Reward:** {current_best_sol.evaluation_res.score:.2f}
**Current Approach:** {current_best_sol.other_info.get("thought", "Simple rule-based controller")}

**Code:**
```python
{current_best_sol.sol_string}
```

## EPISODE ANALYSIS
{episode_analysis}

## OPTIMIZATION INSIGHTS
{insights_text}

## OPTIMIZATION STRATEGY
Analyze the baseline policy and propose improvements:
1. Identify weaknesses in the current control logic
2. Consider better state interpretation methods
3. Design smarter action selection rules
4. Balance fuel efficiency with landing success

## RESPONSE FORMAT
name: [descriptive_name_with_underscores]
code:
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your improved implementation
    return action
```
thought: [Your rationale for the improvement - what specific changes did you make and why?]

## FORMAT REQUIREMENTS
1. The code MUST be wrapped in ```python and ``` markers
2. MAKE SURE THE POLICY CODE IS VALID PYTHON
3. The policy function must return an integer in {{0, 1, 2, 3}}
"""
        return [{"role": "user", "content": prompt}]

    def _get_crossover_prompt(
        self,
        task_description: str,
        current_best_sol: Solution,
        parents: List[Solution],
        episode_analysis: str,
        insights_text: str,
    ) -> List[dict]:
        """Generate crossover prompt."""
        # Format parent information
        parents_info = ""
        for i, parent in enumerate(parents, 1):
            parent_score = parent.evaluation_res.score if parent.evaluation_res else 0
            parents_info += f"""
**Parent {i}:** {parent.other_info.get("name", f"parent_{i}")}
- Average Reward: {parent_score:.2f}
- Approach: {parent.other_info.get("thought", "No description")}
```python
{parent.sol_string}
```
"""

        prompt = f"""# CONTROL POLICY CROSSOVER TASK

{task_description}

## CURRENT BEST POLICY
**Name:** {current_best_sol.other_info.get("name", "current_best")}
**Average Reward:** {current_best_sol.evaluation_res.score:.2f}
```python
{current_best_sol.sol_string}
```

## PARENT POLICIES TO COMBINE
{parents_info}

## EPISODE ANALYSIS (Current Best)
{episode_analysis}

## OPTIMIZATION INSIGHTS
{insights_text}

## CROSSOVER STRATEGY
Combine the best aspects of both parent policies:
1. Identify successful control patterns from each parent
2. Find complementary strategies that can be merged
3. Resolve any conflicts between approaches
4. Create a unified, improved policy

## RESPONSE FORMAT
name: [descriptive_name_with_underscores]
code:
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your hybrid implementation
    return action
```
thought: [Explain which ideas you took from each parent and how you combined them]

## FORMAT REQUIREMENTS
1. The code MUST be wrapped in ```python and ``` markers
2. MAKE SURE THE POLICY CODE IS VALID PYTHON
3. The policy function must return an integer in {{0, 1, 2, 3}}
"""
        return [{"role": "user", "content": prompt}]

    def _get_mutation_prompt(
        self,
        task_description: str,
        current_best_sol: Solution,
        target: Solution,
        episode_analysis: str,
        insights_text: str,
    ) -> List[dict]:
        """Generate mutation prompt."""
        target_score = target.evaluation_res.score if target.evaluation_res else 0

        prompt = f"""# CONTROL POLICY MUTATION TASK

{task_description}

## CURRENT BEST POLICY
**Name:** {current_best_sol.other_info.get("name", "current_best")}
**Average Reward:** {current_best_sol.evaluation_res.score:.2f}
```python
{current_best_sol.sol_string}
```

## POLICY TO MUTATE
**Name:** {target.other_info.get("name", "mutation_target")}
**Average Reward:** {target_score:.2f}
**Approach:** {target.other_info.get("thought", "No description")}
```python
{target.sol_string}
```

## EPISODE ANALYSIS (Current Best)
{episode_analysis}

## OPTIMIZATION INSIGHTS
{insights_text}

## MUTATION STRATEGY
Apply significant modifications to the target policy:
1. Change the control logic structure (not just parameters)
2. Add or remove decision conditions
3. Reorganize the priority of actions
4. Improve handling of specific scenarios

Create a substantially different version that explores new directions.

## RESPONSE FORMAT
name: [descriptive_name_with_underscores]
code:
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    # Your mutated implementation
    return action
```
thought: [Explain what significant changes you made and why they might improve performance]

## FORMAT REQUIREMENTS
1. The code MUST be wrapped in ```python and ``` markers
2. MAKE SURE THE POLICY CODE IS VALID PYTHON
3. The policy function must return an integer in {{0, 1, 2, 3}}
"""
        return [{"role": "user", "content": prompt}]

    def parse_response(self, response_str: str) -> Solution:
        """
        Parse LLM response with multiple fallback strategies.

        Extracts name, code, and thought from structured response.
        """
        if not response_str or not response_str.strip():
            return Solution("")

        content = response_str.strip()

        # Strategy 1: Standard format parsing
        result = self._parse_standard_format(content)
        if result and result[1]:
            return Solution(
                result[1],
                other_info={"name": result[0], "thought": result[2]}
            )

        # Strategy 2: Flexible format parsing
        result = self._parse_flexible_format(content)
        if result and result[1]:
            return Solution(
                result[1],
                other_info={"name": result[0], "thought": result[2]}
            )

        # Strategy 3: Code block extraction only
        code = self._extract_any_code_block(content)
        if code:
            return Solution(
                code,
                other_info={"name": "extracted", "thought": "Fallback parsing"}
            )

        # Strategy 4: Raw content (last resort)
        return Solution(
            content,
            other_info={"name": "raw", "thought": "Failed to parse response"}
        )

    def _parse_standard_format(self, content: str) -> tuple:
        """Parse standard format: name -> code -> thought."""
        # Extract name
        name_pattern = r"^name:\s*([^\n\r]+?)(?:\n|\r|$)"
        name_match = re.search(name_pattern, content, re.MULTILINE | re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else ""

        # Extract code block
        code_pattern = r"code:\s*\n*```(?:python|py)?\n(.*?)```"
        code_match = re.search(code_pattern, content, re.DOTALL | re.IGNORECASE)
        code = code_match.group(1).strip() if code_match else ""

        # Extract thought
        thought_pattern = r"thought:\s*(.*?)$"
        thought_match = re.search(thought_pattern, content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        return (name, code, thought)

    def _parse_flexible_format(self, content: str) -> tuple:
        """More flexible parsing for format variations."""
        # Try to extract name anywhere
        name_pattern = r"(?:name|Name|NAME)\s*:?\s*([^\n\r]+)"
        name_match = re.search(name_pattern, content, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else ""

        # Extract any code block
        code = self._extract_any_code_block(content)

        # Try to extract thought
        thought_pattern = r"(?:thought|Thought|THOUGHT)\s*:?\s*(.*?)(?=\n(?:name|code)|$)"
        thought_match = re.search(thought_pattern, content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        return (name, code, thought)

    def _extract_any_code_block(self, content: str) -> str:
        """Extract any Python code block from content."""
        # Priority 1: ```python blocks
        python_pattern = r"```(?:python|py)\n(.*?)```"
        match = re.search(python_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Priority 2: Any ``` blocks
        generic_pattern = r"```[^\n]*\n(.*?)```"
        match = re.search(generic_pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Priority 3: code: section without markers
        code_pattern = r"code:\s*\n*(.*?)(?=\n(?:thought|$))"
        match = re.search(code_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            code_content = match.group(1).strip()
            code_content = re.sub(r"^```[^\n]*\n?", "", code_content)
            code_content = re.sub(r"\n?```\s*$", "", code_content)
            return code_content.strip()

        return ""
