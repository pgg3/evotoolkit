# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Basic Scientific Regression Example

This example demonstrates how to use EvoToolkit to discover mathematical equations
from real scientific datasets using the bacterial growth dataset.

Requirements:
- pip install evotoolkit[scientific_regression]
- Set your API key in the code or use environment variable
"""

import os
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi


def main():
    print("="*60)
    print("Scientific Symbolic Regression Example")
    print("="*60)

    # Step 1: Create a task
    print("\n[1/4] Creating scientific regression task...")
    task = ScientificRegressionTask(
        dataset_name="bactgrow",  # Bacterial growth dataset
        max_params=10,            # Number of optimizable parameters
        timeout_seconds=60.0      # Timeout per evaluation
    )

    print(f"Dataset: {task.dataset_name}")
    print(f"Training size: {task.task_info['train_size']}")
    print(f"Test size: {task.task_info['test_size']}")

    # Step 2: Test with initial solution
    print("\n[2/4] Testing initial solution...")
    init_sol = task.make_init_sol_wo_other_info()
    result = task.evaluate_code(init_sol.sol_string)
    print(f"Initial solution score: {result.score:.6f}")
    print(f"Initial test MSE: {result.additional_info['test_mse']:.6f}")

    # Step 3: Create interface
    print("\n[3/4] Setting up EvoEngineer interface...")
    interface = EvoEngineerPythonInterface(task)

    # Step 4: Configure LLM API
    print("\n[4/4] Configuring LLM API...")
    llm_api = HttpsApi(
        api_url="https://api.openai.com/v1/chat/completions",
        key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),  # Or set directly: key="sk-..."
        model="gpt-4o"
    )

    # Run evolution
    print("\n" + "="*60)
    print("Starting evolution...")
    print("This may take a few minutes...")
    print("="*60 + "\n")

    result = evotoolkit.solve(
        interface=interface,
        output_path='./scientific_regression_results',
        running_llm=llm_api,
        max_generations=3,  # Reduced for faster testing
        pop_size=5
    )

    # Display results
    print("\n" + "="*60)
    print("Evolution completed!")
    print("="*60)
    print(f"\nBest solution score: {result.evaluation_res.score:.6f}")
    print(f"Results saved to: ./scientific_regression_results/")
    print(f"\nBest equation:\n{result.sol_string}")


if __name__ == "__main__":
    main()
