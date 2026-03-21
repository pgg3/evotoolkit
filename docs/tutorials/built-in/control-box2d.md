# Control Box2D (Lunar Lander) Tutorial

Learn how to use LLM-driven evolution to discover interpretable control policies for the Gymnasium LunarLander-v3 environment.

!!! tip "Complete Example Code"
    This tutorial provides complete, runnable examples (click to view/download):

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/lunar_lander/basic_example.py) - Basic usage example
    - [:material-file-document: README.md](https://github.com/pgg3/evotoolkit/blob/master/examples/lunar_lander/README.md) - Examples documentation and usage guide

    Run locally:
    ```bash
    cd examples/lunar_lander
    python basic_example.py
    ```

---

## Overview

This tutorial demonstrates:

- Creating a LunarLander control task
- Evolving interpretable Python control policies using LLM-driven evolution
- Understanding the `policy(state) -> action` function interface
- Evaluating policies in the Gymnasium environment
- Running evolution to discover effective landing strategies

Unlike neural network controllers, EvoToolkit evolves **human-readable Python code**, enabling inspection, understanding, and formal verification of the resulting policies.

---

## Installation

```bash
pip install evotoolkit[control_box2d]
```

This installs:

- `gymnasium[box2d]` - Gymnasium environment with Box2D physics
- `box2d-py` - Box2D physics engine

**Prerequisites:**

- Python >= 3.10
- LLM API access (OpenAI, Claude, or other compatible providers)

---

## Understanding the LunarLander Task

### What Does the Task Evolve?

The task evolves a `policy` function that maps an 8-dimensional state observation to one of 4 discrete actions:

| State Index | Meaning |
|-------------|---------|
| 0 | X position |
| 1 | Y position |
| 2 | X velocity |
| 3 | Y velocity |
| 4 | Angle |
| 5 | Angular velocity |
| 6 | Left leg contact (bool) |
| 7 | Right leg contact (bool) |

| Action | Meaning |
|--------|---------|
| 0 | Do nothing |
| 1 | Fire left engine |
| 2 | Fire main engine |
| 3 | Fire right engine |

### Evaluation

Each policy is evaluated across multiple episodes. The score is the average reward per episode:

- **Perfect landing**: ~200 points
- **Crash**: -100 points
- **Fuel efficiency**: -0.3 per frame of engine use

---

## Quick Start

### Step 1: Create the Task

```python
from evotoolkit.task.python_task.control_box2d import LunarLanderTask

task = LunarLanderTask(
    num_episodes=5,       # Episodes per evaluation
    max_steps=1000,       # Max steps per episode
    render_mode=None,     # Set to "human" to watch
    seed=42,              # For reproducibility
    timeout_seconds=60.0,
)

print(f"Environment: {task.task_info['env_name']}")
print(f"State dimensions: {task.task_info['state_dim']}")
print(f"Action dimensions: {task.task_info['action_dim']}")
```

### Step 2: Test the Baseline Policy

```python
# Evaluate a baseline policy directly
from evotoolkit.core import Solution

baseline = Solution("def policy(obs):\n    return 0")
result = task.evaluate(baseline)

print(f"Baseline score: {result.score:.2f}")
if result.valid:
    print(f"Avg reward: {result.additional_info['avg_reward']:.2f}")
    print(f"Success rate: {result.additional_info['success_rate']:.1%}")
```

### Step 3: Run Evolution

```python
import evotoolkit
from evotoolkit.task.python_task.control_box2d import EvoEngineerControlInterface
from evotoolkit.tools.llm import HttpsApi

# Create control-specific interface
interface = EvoEngineerControlInterface(task)

# Configure LLM
llm_api = HttpsApi(
    api_url="api.openai.com",
    key="your-api-key-here",
    model="gpt-4o"
)

# Run evolution
result = evotoolkit.solve(
    interface=interface,
    output_path='./lunar_lander_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

print(f"Best policy found:")
print(result.sol_string)
print(f"Score: {result.evaluation_res.score:.2f}")
```

---

## Policy Function Interface

The evolved function must have this exact signature:

```python
def policy(state: list) -> int:
    """
    Control policy for LunarLander-v3.

    Args:
        state: 8-dimensional observation:
            [x_pos, y_pos, x_vel, y_vel, angle, angular_vel,
             left_contact, right_contact]

    Returns:
        action: Integer in {0, 1, 2, 3}
            0 = do nothing
            1 = fire left engine
            2 = fire main engine (upward thrust)
            3 = fire right engine
    """
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
    # Your control logic here
    return 0
```

### Example: Simple Heuristic Policy

```python
def policy(state):
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state

    # Fire main engine if falling fast
    if vy < -1.0:
        return 2

    # Correct angle
    if angle > 0.2:
        return 3  # Fire right to tilt left
    elif angle < -0.2:
        return 1  # Fire left to tilt right

    # Hover and descend slowly
    if vy < -0.5:
        return 2

    return 0
```

---

## Available Interfaces

| Interface | Algorithm | Description |
|-----------|-----------|-------------|
| `EvoEngineerControlInterface` | EvoEngineer | Recommended — uses control-specific prompts |
| `EvoEngineerPythonInterface` | EvoEngineer | Generic Python interface |
| `EoHPythonInterface` | EoH | Heuristic evolution |
| `FunSearchPythonInterface` | FunSearch | Function search |

```python
# Import options
from evotoolkit.task.python_task.control_box2d import EvoEngineerControlInterface
from evotoolkit.task.python_task import EvoEngineerPythonInterface, EoHPythonInterface
```

---

## `LunarLanderTask` API

```python
class LunarLanderTask(PythonTask):
    def __init__(
        self,
        num_episodes: int = 10,       # Episodes per evaluation
        max_steps: int = 1000,        # Max steps per episode
        render_mode: str | None = None,  # "human" to visualize
        use_mock: bool = False,       # Return random score (for testing)
        seed: int | None = None,      # Random seed
        timeout_seconds: float = 60.0,  # Execution timeout
    )
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `evaluate(solution)` | Evaluate a policy candidate, returns `EvaluationResult` |

**`EvaluationResult.additional_info` keys:**

| Key | Description |
|-----|-------------|
| `avg_reward` | Average reward over episodes |
| `std_reward` | Standard deviation of rewards |
| `success_rate` | Fraction of episodes with reward > 200 |
| `min_reward` | Minimum episode reward |
| `max_reward` | Maximum episode reward |

---

## Tips for Better Results

1. **Use more episodes** for stable evaluation: `num_episodes=10` or more
2. **Set a seed** for reproducibility during development
3. **Use `EvoEngineerControlInterface`** — it includes control-specific domain knowledge in prompts
4. **Increase `max_generations`** for harder tasks

---

## Next Steps

- [Customizing Evolution Methods](../customization/customizing-evolution.md) — Modify prompts for better control policies
- [Advanced Usage](../advanced-overview.md) — Low-level API and configuration
- [API Reference](../../api/index.md) — Complete API documentation
