# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
LunarLander Control Policy Evolution Task.

This task evolves interpretable Python control policies for the
Gymnasium LunarLander-v3 environment using LLM-driven code evolution.
"""

import math
from typing import Callable

import numpy as np

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.registry import register_task
from evotoolkit.task.python_task.python_task import PythonTask

# Safe builtins for code execution
SAFE_BUILTINS = {
    # Basic types
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    # Math functions
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "pow": pow,
    # Iteration utilities
    "range": range,
    "len": len,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    # Conditionals
    "all": all,
    "any": any,
    # Type checking
    "isinstance": isinstance,
    "type": type,
    # Debug
    "print": print,
    # Import (restricted)
    "__import__": __import__,
}


@register_task("LunarLander")
class LunarLanderTask(PythonTask):
    """
    LunarLander control policy evolution task.

    This task evolves a Python function `policy(state) -> action` that
    controls the lunar lander to safely land on the landing pad.

    The evolved policy is human-readable Python code, not a black-box
    neural network, enabling interpretation and formal verification.
    """

    def __init__(
        self,
        num_episodes: int = 10,
        max_steps: int = 1000,
        render_mode: str | None = None,
        use_mock: bool = False,
        seed: int | None = None,
        timeout_seconds: float = 60.0,
    ):
        """
        Initialize LunarLander task.

        Args:
            num_episodes: Number of episodes to run for evaluation.
                More episodes give more stable fitness estimates but
                take longer. Recommended: 3-5 for development, 10-20
                for final evaluation.
            max_steps: Maximum steps per episode. LunarLander typically
                terminates within 1000 steps.
            render_mode: Gymnasium render mode. Use "human" for visual
                debugging, None for faster training.
            use_mock: If True, returns random fitness without running
                the actual environment. Useful for testing.
            seed: Random seed for reproducibility. If set, episode i
                uses seed (seed + i).
            timeout_seconds: Maximum time for code execution.
        """
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.use_mock = use_mock
        self.seed = seed

        data = {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "render_mode": render_mode,
            "use_mock": use_mock,
            "seed": seed,
        }

        super().__init__(data=data, timeout_seconds=timeout_seconds)

    def _process_data(self, data):
        """Process input data and create task_info."""
        self.data = data
        self.task_info = {
            "task_name": "LunarLander Control",
            "env_name": "LunarLander-v3",
            "state_dim": 8,
            "action_dim": 4,
            "num_episodes": data["num_episodes"],
            "max_steps": data["max_steps"],
        }

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """
        Evaluate Python code for LunarLander control.

        The code must define a `policy` function that takes an 8-dim
        state array and returns an integer action in {0, 1, 2, 3}.
        """
        # Create safe execution namespace
        namespace = {
            "__builtins__": SAFE_BUILTINS,
            "np": np,
            "numpy": np,
            "math": math,
        }

        # Execute the candidate code
        exec(candidate_code, namespace)

        # Check if policy function exists
        if "policy" not in namespace:
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": 'Function "policy" not found in code'},
            )

        policy_func = namespace["policy"]

        # Validate it's callable
        if not callable(policy_func):
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": '"policy" is not a callable function'},
            )

        # Mock evaluation for testing
        if self.use_mock:
            return self._mock_evaluate()

        # Real evaluation in LunarLander environment
        return self._evaluate_policy(policy_func)

    def _mock_evaluate(self) -> EvaluationResult:
        """Return mock evaluation result for testing."""
        mock_score = np.random.uniform(-200, 300)
        return EvaluationResult(
            valid=True,
            score=mock_score,
            additional_info={
                "avg_reward": mock_score,
                "std_reward": np.random.uniform(20, 80),
                "min_reward": mock_score - np.random.uniform(50, 150),
                "max_reward": mock_score + np.random.uniform(50, 150),
                "avg_length": np.random.uniform(200, 800),
                "mock": True,
            },
        )

    def _evaluate_policy(self, policy: Callable) -> EvaluationResult:
        """
        Evaluate a policy function in the LunarLander environment.

        Args:
            policy: Function that takes state array and returns action.

        Returns:
            EvaluationResult with average reward as score.
        """
        import gymnasium as gym

        env = gym.make("LunarLander-v3", render_mode=self.render_mode)

        total_rewards = []
        episode_lengths = []
        landing_success = []

        for episode in range(self.num_episodes):
            # Set seed for reproducibility
            episode_seed = (self.seed + episode) if self.seed is not None else None
            state, _ = env.reset(seed=episode_seed)

            episode_reward = 0.0
            terminated = False
            truncated = False

            for step in range(self.max_steps):
                # Call the evolved policy
                try:
                    action = policy(state)
                except Exception as e:
                    env.close()
                    return EvaluationResult(
                        valid=False,
                        score=float("-inf"),
                        additional_info={
                            "error": f"Policy execution error at step {step}: {str(e)}",
                            "episode": episode,
                        },
                    )

                # Validate action
                if not isinstance(action, (int, np.integer)):
                    # Try to convert to int if possible
                    try:
                        action = int(action)
                    except (TypeError, ValueError):
                        env.close()
                        return EvaluationResult(
                            valid=False,
                            score=float("-inf"),
                            additional_info={
                                "error": f"Invalid action type: {type(action).__name__}, expected int",
                                "action": str(action),
                            },
                        )

                if action not in [0, 1, 2, 3]:
                    env.close()
                    return EvaluationResult(
                        valid=False,
                        score=float("-inf"),
                        additional_info={
                            "error": f"Invalid action value: {action}, must be in {{0, 1, 2, 3}}",
                        },
                    )

                # Step environment
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            total_rewards.append(episode_reward)
            episode_lengths.append(step + 1)

            # Check if landed successfully (reward > 100 typically indicates success)
            landing_success.append(episode_reward > 100)

        env.close()

        # Compute statistics
        avg_reward = float(np.mean(total_rewards))
        std_reward = float(np.std(total_rewards))
        min_reward = float(np.min(total_rewards))
        max_reward = float(np.max(total_rewards))
        avg_length = float(np.mean(episode_lengths))
        success_rate = float(np.mean(landing_success))

        # Check for invalid scores
        if np.isnan(avg_reward) or np.isinf(avg_reward):
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": "Invalid reward (NaN or Inf)"},
            )

        return EvaluationResult(
            valid=True,
            score=avg_reward,  # Higher reward = better policy
            additional_info={
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "min_reward": min_reward,
                "max_reward": max_reward,
                "avg_length": avg_length,
                "success_rate": success_rate,
                "all_rewards": total_rewards,
                "all_lengths": episode_lengths,
            },
        )

    def get_base_task_description(self) -> str:
        """Get task description for LLM prompt generation."""
        return """## Task: LunarLander Control Policy

Design a control policy function for the Lunar Lander task. The lander starts at the
top center with random initial force. Your goal is to land safely on the landing pad
(between the two flags at coordinates (0, 0)).

### State Space (8-dimensional numpy array)
The `state` parameter is a numpy array with 8 elements:
- `state[0]`: x position (horizontal, 0 = center of landing pad)
- `state[1]`: y position (vertical height, 0 = ground level)
- `state[2]`: x velocity (horizontal speed, positive = moving right)
- `state[3]`: y velocity (vertical speed, negative = falling down)
- `state[4]`: angle (in radians, 0 = upright, positive = tilted right)
- `state[5]`: angular velocity (positive = rotating clockwise)
- `state[6]`: left leg contact (1.0 if touching ground, else 0.0)
- `state[7]`: right leg contact (1.0 if touching ground, else 0.0)

### Action Space (discrete integer 0-3)
Return one of four actions:
- `0`: Do nothing (let gravity and momentum continue)
- `1`: Fire left engine (pushes lander to the RIGHT)
- `2`: Fire main engine (pushes lander UP, slows descent)
- `3`: Fire right engine (pushes lander to the LEFT)

### Reward Structure
- Landing on pad: +100 to +140 points (closer to center = higher)
- Each leg ground contact: +10 points
- Crash (too fast or bad angle): -100 points
- Main engine firing: -0.3 per frame (fuel cost)
- Side engine firing: -0.03 per frame (fuel cost)
- Moving away from landing zone: penalty proportional to distance

**Success criterion**: Average reward >= 200 over 100 episodes

### Function Signature
```python
import numpy as np

def policy(state: np.ndarray) -> int:
    \"\"\"
    Determine the action based on current state.

    Args:
        state: numpy array of shape (8,) containing lander state

    Returns:
        action: integer in {0, 1, 2, 3}
    \"\"\"
    # Your implementation here
    return action
```

### Control Strategy Tips
1. **Vertical control**: Fire main engine (2) when falling too fast (vy < -0.5)
2. **Horizontal control**: Use side engines to move toward landing pad center
   - If drifting right (x > 0), fire right engine (3) to push left
   - If drifting left (x < 0), fire left engine (1) to push right
3. **Attitude control**: Keep the lander upright
   - If tilted right (angle > 0), fire right engine (3)
   - If tilted left (angle < 0), fire left engine (1)
4. **Fuel efficiency**: Only fire engines when necessary
5. **Landing**: Both legs should touch ground gently for a successful landing
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial solution with a simple rule-based policy."""
        baseline_code = '''import numpy as np

def policy(state: np.ndarray) -> int:
    """
    Simple rule-based policy for LunarLander.

    Strategy:
    1. Fire main engine if falling too fast
    2. Use side engines to correct horizontal position and angle
    """
    # Unpack state for readability
    x = state[0]          # horizontal position
    y = state[1]          # vertical position
    vx = state[2]         # horizontal velocity
    vy = state[3]         # vertical velocity
    angle = state[4]      # angle (radians)
    angular_vel = state[5]  # angular velocity
    left_leg = state[6]   # left leg contact
    right_leg = state[7]  # right leg contact

    # Priority 1: Prevent crash by slowing descent
    if vy < -0.5 and y > 0.1:
        return 2  # Fire main engine

    # Priority 2: Correct dangerous tilt
    if angle > 0.2 or (angle > 0.05 and angular_vel > 0.1):
        return 3  # Fire right engine (rotate left)
    if angle < -0.2 or (angle < -0.05 and angular_vel < -0.1):
        return 1  # Fire left engine (rotate right)

    # Priority 3: Move toward landing pad (x = 0)
    if x > 0.2 and vx > -0.1:
        return 3  # Fire right engine (push left)
    if x < -0.2 and vx < 0.1:
        return 1  # Fire left engine (push right)

    # Priority 4: Slow down if moving too fast horizontally
    if vx > 0.3:
        return 3  # Fire right engine
    if vx < -0.3:
        return 1  # Fire left engine

    # Default: Do nothing to save fuel
    return 0
'''

        # Evaluate the initial solution
        eval_res = self.evaluate_code(baseline_code)

        return Solution(sol_string=baseline_code, evaluation_res=eval_res, other_info={})
