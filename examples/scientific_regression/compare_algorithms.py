# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Compare Different Evolution Algorithms

This example compares three evolutionary algorithms on the same task:
- EvoEngineer: LLM-driven evolution with init, mutation, and crossover operators
- EoH: Evolution of Heuristics
- FunSearch: Function search optimization

Requirements:
- pip install evotoolkit[scientific_regression]
- Set your API key in the code or use environment variable
"""

import os
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import (
    EvoEngineerPythonInterface,
    EoHPythonInterface,
    FunSearchPythonInterface
)
from evotoolkit.tools.llm import HttpsApi


def run_algorithm(algorithm_name, interface_class, task, llm_api, output_dir):
    """Run a single evolutionary algorithm and return results."""
    print(f"\n{'='*60}")
    print(f"Running {algorithm_name}...")
    print(f"{'='*60}\n")

    # Create interface
    interface = interface_class(task)

    # Run evolution
    result = evotoolkit.solve(
        interface=interface,
        output_path=output_dir,
        running_llm=llm_api,
        max_generations=3,
        pop_size=5
    )

    print(f"\n{algorithm_name} completed!")
    print(f"Best score: {result.evaluation_res.score:.6f}")

    return {
        'algorithm': algorithm_name,
        'score': result.evaluation_res.score,
        'solution': result.sol_string,
        'output_path': output_dir
    }


def main():
    print("="*60)
    print("Comparing Evolution Algorithms")
    print("="*60)

    # Create task (shared by all algorithms)
    print("\n[1/2] Creating scientific regression task...")
    task = ScientificRegressionTask(
        dataset_name="bactgrow",
        max_params=10,
        timeout_seconds=60.0
    )
    print(f"Dataset: {task.dataset_name}")
    print(f"Training size: {task.task_info['train_size']}")
    print(f"Test size: {task.task_info['test_size']}")

    # Configure LLM (shared by all algorithms)
    print("\n[2/2] Configuring LLM API...")
    llm_api = HttpsApi(
        api_url="https://api.openai.com/v1/chat/completions",
        key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        model="gpt-4o"
    )

    # Define algorithms to compare
    algorithms = [
        ("EvoEngineer", EvoEngineerPythonInterface, "./results_evoengineer"),
        ("EoH", EoHPythonInterface, "./results_eoh"),
        ("FunSearch", FunSearchPythonInterface, "./results_funsearch")
    ]

    # Run all algorithms
    results = []
    for algo_name, interface_class, output_dir in algorithms:
        result = run_algorithm(algo_name, interface_class, task, llm_api, output_dir)
        results.append(result)

    # Display comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    # Sort by score (lower is better for MSE)
    results.sort(key=lambda x: x['score'])

    print("\nRanking (lower score is better):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['algorithm']}")
        print(f"   Score: {result['score']:.6f}")
        print(f"   Output: {result['output_path']}")

    print("\n" + "="*60)
    print(f"Winner: {results[0]['algorithm']} with score {results[0]['score']:.6f}")
    print("="*60)

    print(f"\nBest equation from {results[0]['algorithm']}:")
    print(results[0]['solution'])


if __name__ == "__main__":
    main()
