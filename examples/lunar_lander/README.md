# LunarLander Control Policy Evolution

This example demonstrates how to evolve interpretable control policies for the Gymnasium LunarLander-v3 environment using LLM-driven code evolution.

## Installation

```bash
pip install evotoolkit[control_box2d]
```

## Quick Start

### 1. Basic Example

Run a basic evolution experiment:

```bash
python basic_example.py
```

This will:
1. Create a LunarLander task
2. Test the baseline policy
3. Evolve better policies using LLM
4. Save results to `./lunar_lander_results/`

### 2. Watch the Policy

Visualize the best evolved policy:

```bash
python watch_policy.py
```

## Examples

| File | Description |
|------|-------------|
| `basic_example.py` | Basic evolution with default settings |
| `watch_policy.py` | Visualize and evaluate a policy |

## Task Details

### State Space (8 dimensions)

| Index | Name | Description |
|-------|------|-------------|
| 0 | x | Horizontal position (0 = landing pad center) |
| 1 | y | Vertical height (0 = ground) |
| 2 | vx | Horizontal velocity |
| 3 | vy | Vertical velocity (negative = falling) |
| 4 | angle | Angle in radians (0 = upright) |
| 5 | angular_vel | Angular velocity |
| 6 | left_leg | Left leg ground contact (0 or 1) |
| 7 | right_leg | Right leg ground contact (0 or 1) |

### Action Space (4 discrete actions)

| Action | Effect |
|--------|--------|
| 0 | Do nothing |
| 1 | Fire left engine (push right) |
| 2 | Fire main engine (push up) |
| 3 | Fire right engine (push left) |

### Reward Structure

- Landing on pad: +100 to +140
- Crash: -100
- Leg contact: +10 each
- Main engine: -0.3/frame
- Side engine: -0.03/frame

**Success**: Average reward ≥ 200 over 100 episodes

## Expected Results

| Policy Type | Expected Score |
|-------------|----------------|
| Random | -200 to -100 |
| Baseline (rule-based) | 50 to 150 |
| Evolved (simple) | 150 to 250 |
| Evolved (advanced) | 250+ |

## Tips

1. **Development**: Use `num_episodes=3` for faster iteration
2. **Final evaluation**: Use `num_episodes=20` for stable scores
3. **Debugging**: Set `render_mode="human"` to watch the lander
4. **Reproducibility**: Set `seed=42` for consistent results
