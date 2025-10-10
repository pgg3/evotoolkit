# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Basic Prompt Optimization Example

This example demonstrates how to use EvoToolkit to optimize prompt templates
for better LLM performance on math questions.

Requirements:
- pip install evotoolkit
- Configure API credentials directly in code (or use mock mode)
"""

from evotoolkit.task import PromptOptimizationTask


def main():
    print("=" * 60)
    print("Prompt Optimization Example")
    print("=" * 60)

    # Step 1: Define test cases
    print("\n[1/4] Defining test cases...")

    test_cases = [
        {"question": "What is 2+2?", "expected": "4"},
        {"question": "What is 5*3?", "expected": "15"},
        {"question": "What is 10-7?", "expected": "3"},
        {"question": "What is 12/4?", "expected": "3"},
        {"question": "What is 7+8?", "expected": "15"},
    ]

    print(f"Created {len(test_cases)} test cases")

    # Step 2: Create task (with mock LLM for testing)
    print("\n[2/4] Creating prompt optimization task...")

    # Option 1: Use mock mode (no LLM needed, good for testing)
    task = PromptOptimizationTask(
        test_cases=test_cases,
        use_mock=True,  # Set to False to use real LLM
    )

    # Option 2: Use real LLM (uncomment and configure)
    # llm_api = HttpsApi(
    #     api_url="api.openai.com",  # Your API URL
    #     key="your-api-key-here",   # Your API key
    #     model="gpt-4o"
    # )
    # task = PromptOptimizationTask(
    #     test_cases=test_cases,
    #     llm_api=llm_api,
    #     use_mock=False
    # )

    print(f"Task created with {task.task_info['num_test_cases']} test cases")

    # Step 3: Test initial solution
    print("\n[3/4] Testing initial solution...")

    init_sol = task.make_init_sol_wo_other_info()
    print("Initial prompt template:")
    print(f'  "{init_sol.sol_string}"')
    print(f"Initial score: {init_sol.evaluation_res.score:.2%}")
    print(
        f"Correct: {init_sol.evaluation_res.additional_info['correct']}/{init_sol.evaluation_res.additional_info['total']}"
    )

    # Step 4: Test a custom prompt template
    print("\n[4/4] Testing custom prompt template...")

    custom_template = "Solve this math problem and give only the number: {question}"
    result = task.evaluate_code(custom_template)
    print(f'Custom template: "{custom_template}"')
    print(f"Score: {result.score:.2%}")
    print(
        f"Correct: {result.additional_info['correct']}/{result.additional_info['total']}"
    )

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey difference from Python task:")
    print("- Solutions are STRING TEMPLATES (not Python code)")
    print("- Templates use {question} placeholder")
    print("- Evolution optimizes the prompt string directly")

    print("\nTo run evolution:")
    print("1. Configure LLM API credentials in the code above")
    print("2. Set use_mock=False to use real LLM")
    print("3. Run evolution (example below):")
    print("""
    interface = EvoEngineerStringInterface(task)
    llm_api = HttpsApi(
        api_url="api.openai.com",  # Your API URL
        key="your-api-key-here",   # Your API key
        model="gpt-4o"
    )
    result = evotoolkit.solve(
        interface=interface,
        output_path='./results',
        running_llm=llm_api,
        max_generations=5
    )
    """)


if __name__ == "__main__":
    main()
