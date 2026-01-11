# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Basic LunarLander Control Policy Evolution Example

This example demonstrates how to use EvoToolkit to evolve interpretable
control policies for the Gymnasium LunarLander-v3 environment.

Requirements:
- pip install evotoolkit[control_box2d]
- Set your API key in environment variable or in the code
"""

import os

import evotoolkit
from evotoolkit.task.python_task.control_box2d import (
    EvoEngineerControlInterface,
    LunarLanderTask,
)
from evotoolkit.tools.llm import HttpsApi


def main():
    print("=" * 60)
    print("LunarLander Control Policy Evolution")
    print("=" * 60)

    # Step 1: Create the task
    print("\n[1/4] Creating LunarLander task...")
    task = LunarLanderTask(
        num_episodes=5,  # Episodes per evaluation (use 10-20 for final)
        max_steps=1000,  # Max steps per episode
        render_mode=None,  # Set to "human" to watch
        seed=42,  # For reproducibility
        timeout_seconds=60.0,
    )

    print(f"Environment: {task.task_info['env_name']}")
    print(f"State dimensions: {task.task_info['state_dim']}")
    print(f"Action dimensions: {task.task_info['action_dim']}")
    print(f"Episodes per evaluation: {task.task_info['num_episodes']}")

    # Step 2: Test baseline solution
    print("\n[2/4] Testing baseline solution...")
    init_sol = task.make_init_sol_wo_other_info()
    result = task.evaluate_code(init_sol.sol_string)

    print(f"Baseline score: {result.score:.2f}")
    if result.valid:
        print(f"  Avg reward: {result.additional_info['avg_reward']:.2f}")
        print(f"  Std reward: {result.additional_info['std_reward']:.2f}")
        print(f"  Success rate: {result.additional_info['success_rate']:.1%}")
    else:
        print(f"  Error: {result.additional_info.get('error', 'Unknown')}")

    # Step 3: Create control-specific interface
    print("\n[3/4] Setting up EvoEngineerControl interface...")
    interface = EvoEngineerControlInterface(task)

    # Step 4: Configure LLM API
    print("\n[4/4] Configuring LLM API...")
    llm_api = HttpsApi(
        api_url="https://api.openai.com/v1/chat/completions",
        key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        model="gpt-4o",
    )

    # Run evolution
    print("\n" + "=" * 60)
    print("Starting evolution...")
    print("This may take several minutes...")
    print("=" * 60 + "\n")

    best_solution = evotoolkit.solve(
        interface=interface,
        output_path="./lunar_lander_results",
        running_llm=llm_api,
        max_generations=5,  # Increase for better results
        pop_size=5,
    )

    # Display results
    print("\n" + "=" * 60)
    print("Evolution completed!")
    print("=" * 60)

    print(f"\nBest solution score: {best_solution.evaluation_res.score:.2f}")
    if best_solution.evaluation_res.valid:
        info = best_solution.evaluation_res.additional_info
        print(f"  Avg reward: {info['avg_reward']:.2f}")
        print(f"  Success rate: {info['success_rate']:.1%}")

    print("\nResults saved to: ./lunar_lander_results/")
    print("\nBest policy code:")
    print("-" * 40)
    print(best_solution.sol_string)
    print("-" * 40)

    # Optional: Watch the best policy
    print("\nTo watch the best policy, run:")
    print("  python watch_policy.py")


if __name__ == "__main__":
    main()
