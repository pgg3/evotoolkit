# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Custom Prompt Example for Scientific Regression

This example shows how to customize the evolution process by:
1. Inheriting from EvoEngineerPythonInterface
2. Overriding get_operator_prompt() to customize prompts for specific operators
3. Emphasizing physical/biological principles in the prompt

Requirements:
- pip install evotoolkit[scientific_regression]
- Set your API key in the code or use environment variable
"""

import os
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi


class ScientificRegressionInterface(EvoEngineerPythonInterface):
    """Custom Interface optimized for scientific equation discovery.

    This interface customizes the mutation operator prompt to emphasize
    physical and biological principles.
    """

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """Customize prompt for mutation operator to emphasize scientific principles."""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            prompt = f"""You are an expert in scientific equation discovery.

Task: {task_description}

Current best equation (score: {current_best_sol.evaluation_res.score:.5f}):
{current_best_sol.sol_string}

Requirements: Generate an improved equation based on known physical/biological principles:
- Monod equation for substrate limitation
- Arrhenius equation for temperature effects
- Gaussian functions for optimal pH/temperature ranges
- Logistic growth with carrying capacity

Ensure numerical stability and model parsimony.

Output format:
- name: equation name
- code: Python code implementing the equation
- thought: improvement rationale based on scientific principles
"""
            return [{"role": "user", "content": prompt}]

        # Use default prompts for init and crossover operators
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)


def main():
    print("="*60)
    print("Custom Prompt Scientific Regression Example")
    print("="*60)

    # Create task
    print("\n[1/3] Creating scientific regression task...")
    task = ScientificRegressionTask(
        dataset_name="bactgrow",
        max_params=10,
        timeout_seconds=60.0
    )
    print(f"Dataset: {task.dataset_name}")

    # Create custom interface
    print("\n[2/3] Setting up custom interface...")
    interface = ScientificRegressionInterface(task)
    print("Using custom prompt that emphasizes scientific principles")

    # Configure LLM
    print("\n[3/3] Configuring LLM API...")
    llm_api = HttpsApi(
        api_url="https://api.openai.com/v1/chat/completions",
        key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        model="gpt-4o"
    )

    # Run evolution
    print("\n" + "="*60)
    print("Starting evolution with custom prompts...")
    print("="*60 + "\n")

    result = evotoolkit.solve(
        interface=interface,
        output_path='./custom_prompt_results',
        running_llm=llm_api,
        max_generations=3,
        pop_size=5
    )

    # Display results
    print("\n" + "="*60)
    print("Evolution completed!")
    print("="*60)
    print(f"\nBest solution score: {result.evaluation_res.score:.6f}")
    print(f"Results saved to: ./custom_prompt_results/")
    print(f"\nBest equation:\n{result.sol_string}")


if __name__ == "__main__":
    main()
