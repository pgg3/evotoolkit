# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Watch and Evaluate LunarLander Policy

This script allows you to visualize and evaluate a control policy
for the LunarLander environment.

Usage:
    python watch_policy.py                    # Watch baseline policy
    python watch_policy.py --episodes 10      # Run 10 episodes
    python watch_policy.py --no-render        # Evaluate without rendering
"""

import argparse

import numpy as np

from evotoolkit.task.python_task.control_box2d import LunarLanderTask


# You can paste your evolved policy here
CUSTOM_POLICY = """
import numpy as np

def policy(state: np.ndarray) -> int:
    # Unpack state
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state

    # Priority 1: Slow descent
    if vy < -0.5 and y > 0.1:
        return 2

    # Priority 2: Correct tilt
    if angle > 0.2 or (angle > 0.05 and angular_vel > 0.1):
        return 3
    if angle < -0.2 or (angle < -0.05 and angular_vel < -0.1):
        return 1

    # Priority 3: Center horizontally
    if x > 0.2 and vx > -0.1:
        return 3
    if x < -0.2 and vx < 0.1:
        return 1

    # Priority 4: Reduce horizontal speed
    if vx > 0.3:
        return 3
    if vx < -0.3:
        return 1

    return 0
"""


def watch_policy(policy_code: str, num_episodes: int = 5, render: bool = True):
    """
    Watch and evaluate a policy.

    Args:
        policy_code: Python code defining a policy(state) function
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    import gymnasium as gym

    # Create environment
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    # Compile policy
    namespace = {"np": np, "numpy": np}
    exec(policy_code, namespace)
    policy = namespace["policy"]

    # Run episodes
    rewards = []
    lengths = []

    print(f"\nRunning {num_episodes} episodes...")
    print("-" * 40)

    for episode in range(num_episodes):
        state, _ = env.reset(seed=episode)
        episode_reward = 0

        for step in range(1000):
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        lengths.append(step + 1)
        status = "SUCCESS" if episode_reward > 100 else "FAIL"
        print(f"Episode {episode + 1}: reward = {episode_reward:.1f}, "
              f"steps = {step + 1}, {status}")

    env.close()

    # Print summary
    print("-" * 40)
    print(f"\nSummary over {num_episodes} episodes:")
    print(f"  Average reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Min/Max reward: {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"  Average length: {np.mean(lengths):.1f} steps")
    print(f"  Success rate: {np.mean([r > 100 for r in rewards]):.1%}")

    return rewards


def main():
    parser = argparse.ArgumentParser(
        description="Watch and evaluate LunarLander policies"
    )
    parser.add_argument(
        "--episodes", "-n", type=int, default=5,
        help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable rendering for faster evaluation"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Use the baseline policy from LunarLanderTask"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("LunarLander Policy Viewer")
    print("=" * 50)

    if args.baseline:
        print("\nUsing baseline policy from LunarLanderTask...")
        task = LunarLanderTask(num_episodes=1, use_mock=True)
        init_sol = task.make_init_sol_wo_other_info()
        policy_code = init_sol.sol_string
    else:
        print("\nUsing custom policy (edit CUSTOM_POLICY in script to change)...")
        policy_code = CUSTOM_POLICY

    print("\nPolicy code:")
    print("-" * 40)
    print(policy_code)
    print("-" * 40)

    watch_policy(
        policy_code=policy_code,
        num_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
