# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Prompt optimization task implementation.

This task optimizes prompt templates to improve LLM performance on a given task.
"""

import re
from typing import List, Dict, Optional, Any
from evotoolkit.task.string_optimization.string_task import StringTask
from evotoolkit.core import Solution, EvaluationResult


class PromptOptimizationTask(StringTask):
    """
    Task for optimizing LLM prompt templates.

    This task evaluates prompt templates by testing them on a set of test cases
    and measuring the quality of LLM responses.

    The prompt template is a string that can contain {question} placeholder.

    Example:
        >>> # Define test cases
        >>> test_cases = [
        ...     {"question": "What is 2+2?", "expected": "4"},
        ...     {"question": "What is 5*3?", "expected": "15"}
        ... ]
        >>>
        >>> # Create task
        >>> task = PromptOptimizationTask(
        ...     test_cases=test_cases,
        ...     llm_api=my_llm_api
        ... )
        >>>
        >>> # Evaluate a prompt template
        >>> prompt_template = "Solve this math problem: {question}\\nGive only the number."
        >>> result = task.evaluate_code(prompt_template)
    """

    def __init__(
        self,
        test_cases: List[Dict[str, str]],
        llm_api: Optional[Any] = None,
        timeout_seconds: float = 30.0,
        use_mock: bool = False
    ):
        """
        Initialize the prompt optimization task.

        Args:
            test_cases: List of test cases with 'question' and 'expected' keys
            llm_api: LLM API instance for testing prompts (optional if use_mock=True)
            timeout_seconds: Timeout for evaluation
            use_mock: If True, use mock LLM responses for testing (default: False)
        """
        self.test_cases = test_cases
        self.llm_api = llm_api
        self.use_mock = use_mock

        if not use_mock and llm_api is None:
            raise ValueError("llm_api must be provided when use_mock=False")

        data = {
            'test_cases': test_cases,
            'num_cases': len(test_cases)
        }

        super().__init__(data, timeout_seconds)

    def _process_data(self, data):
        """Process task data and set up task_info."""
        super()._process_data(data)
        self.task_info = {
            'num_test_cases': data['num_cases'],
            'task_type': 'prompt_optimization'
        }

    def _evaluate_string_impl(self, prompt_template: str) -> EvaluationResult:
        """
        Evaluate a prompt template string.

        The prompt template should contain {question} placeholder.

        Args:
            prompt_template: Prompt template string with {question} placeholder

        Returns:
            EvaluationResult with score based on correctness rate
        """
        # Validate template has {question} placeholder
        if '{question}' not in prompt_template:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Prompt template must contain {question} placeholder'}
            )

        # Test the prompt on all test cases
        correct = 0
        total = len(self.test_cases)
        results = []

        for case in self.test_cases:
            question = case['question']
            expected = case['expected']

            try:
                # Generate prompt from template
                prompt = prompt_template.format(question=question)

                # Get LLM response
                if self.use_mock:
                    # Mock response: extract number from question for testing
                    response = self._mock_llm_response(question, prompt)
                else:
                    # Real LLM call
                    response = self._call_llm(prompt)

                # Check if answer is correct
                is_correct = self._check_answer(response, expected)

                if is_correct:
                    correct += 1

                results.append({
                    'question': question,
                    'prompt': prompt,
                    'response': response,
                    'expected': expected,
                    'correct': is_correct
                })

            except Exception as e:
                results.append({
                    'question': question,
                    'error': str(e),
                    'correct': False
                })

        # Calculate score as correctness rate
        score = correct / total if total > 0 else 0.0

        return EvaluationResult(
            valid=True,
            score=score,
            additional_info={
                'correct': correct,
                'total': total,
                'accuracy': score,
                'results': results
            }
        )

    def _mock_llm_response(self, question: str, prompt: str) -> str:
        """
        Generate mock LLM response for testing without actual LLM calls.

        Extracts the answer from the question for simple math problems.
        """
        # Simple pattern matching for testing
        # Extract numbers and operators from question
        match = re.search(r'(\d+)\s*([+\-*/])\s*(\d+)', question)
        if match:
            num1, op, num2 = match.groups()
            num1, num2 = int(num1), int(num2)

            ops = {'+': num1 + num2, '-': num1 - num2, '*': num1 * num2, '/': num1 // num2}
            result = ops.get(op, 0)

            # Return in a format that simulates LLM output
            return f"The answer is {result}"

        return "I don't know"

    def _call_llm(self, prompt: str) -> str:
        """Call the real LLM API with the generated prompt."""
        messages = [{"role": "user", "content": prompt}]
        response, usage = self.llm_api.get_response(messages)
        return response

    def _check_answer(self, response: str, expected: str) -> bool:
        """
        Check if LLM response contains the expected answer.

        Args:
            response: LLM response text
            expected: Expected answer string

        Returns:
            True if answer is correct, False otherwise
        """
        # Extract numbers from response
        numbers = re.findall(r'\d+', response)
        return expected in numbers or expected in response

    def get_base_task_description(self) -> str:
        """Get task description for prompt generation."""
        case_desc = "\n".join([
            f"  - Question: {case['question']} | Expected: {case['expected']}"
            for case in self.test_cases[:3]  # Show first 3 examples
        ])

        return f"""# Prompt Optimization Task

**Objective:** Create a prompt template that generates effective prompts for an LLM
to answer questions correctly.

**Test Cases** ({len(self.test_cases)} total, showing first 3):
{case_desc}

**Prompt Template Format:**
Your solution should be a string template with {{question}} placeholder.

Example:
```
Solve this math problem: {{question}}
Give only the number as your answer.
```

**Evaluation:** Your prompt template will be tested on {len(self.test_cases)} questions.
Score = (number of correct answers) / (total questions)

**Tips:**
- Make prompts clear and specific
- Use the {{question}} placeholder to insert the question
- Consider instructing the LLM to output only the answer
- Test different prompt formats for better accuracy
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create an initial solution with a simple prompt template."""
        init_template = "Answer this question: {question}"

        eval_res = self.evaluate_code(init_template)

        return Solution(
            sol_string=init_template,
            evaluation_res=eval_res,
            other_info={}
        )
