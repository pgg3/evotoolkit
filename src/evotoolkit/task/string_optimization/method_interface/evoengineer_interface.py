# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import json
import re
from typing import List

from evotoolkit.core import EvoEngineerInterface, Operator, Solution
from evotoolkit.task.string_optimization.string_task import StringTask


class EvoEngineerStringInterface(EvoEngineerInterface):
    """EvoEngineer Adapter for string optimization tasks.

    This class provides EvoEngineer algorithm logic for string-based tasks
    like prompt optimization, where solutions are strings rather than code.
    """

    def __init__(self, task: StringTask):
        super().__init__(task)

    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for string optimization"""
        return [Operator("init", 0)]

    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for string optimization"""
        return [Operator("crossover", 2), Operator("mutation", 1)]

    def get_operator_prompt(
        self,
        operator_name: str,
        selected_individuals: List[Solution],
        current_best_sol: Solution,
        random_thoughts: List[str],
        **kwargs,
    ) -> List[dict]:
        """Generate prompt for any operator"""
        task_description = self.task.get_base_task_description()

        if operator_name == "init":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join(
                    [f"- {thought}" for thought in random_thoughts]
                )
                thoughts_section = f"""

Reference insights (consider if relevant):
{thoughts_list}

"""

            # For init operator, current_best_sol may be None
            if current_best_sol is not None and current_best_sol.evaluation_res is not None:
                current_solution_section = f"""
Here is the current best solution:

<current_solution>
<string>{current_best_sol.sol_string}</string>
<score>{current_best_sol.evaluation_res.score:.5f}</score>
</current_solution>"""
            else:
                current_solution_section = ""

            prompt = f"""
{task_description}
{current_solution_section}{thoughts_section}

Think deeply about how to {"improve this solution" if current_best_sol else "create a solution"}. {"Reference insights are provided above - use them as inspiration if they seem relevant to your optimization approach." if random_thoughts and len(random_thoughts) > 0 else ""} Propose a {"new solution that" if current_best_sol else "solution that"}:
1. {"Analyzes the current solution to identify improvement opportunities" if current_best_sol else "Addresses the task requirements effectively"}
2. Applies proven techniques and principles
3. Explains your rationale clearly

MAKE SURE THE PROPOSED SOLUTION FOLLOWS THE REQUIRED FORMAT.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
DO NOT USE JSON FORMAT. USE THE SIMPLE KEY-VALUE FORMAT SHOWN BELOW.

Answer using the following schema:

name: A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
solution: The proposed solution as a string.
thought: The rationale for the improvement idea.
"""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "mutation":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join(
                    [f"- {thought}" for thought in random_thoughts]
                )
                thoughts_section = f"""

Mutation insights (consider if relevant):
{thoughts_list}

"""

            individual = selected_individuals[0]

            prompt = f"""
{task_description}

Current best solution (Score: {current_best_sol.evaluation_res.score:.5f}):
{current_best_sol.sol_string}

Solution to mutate (Score: {individual.evaluation_res.score:.5f}):
{individual.sol_string}{thoughts_section}

Create a substantially modified version of the solution to mutate. {"Use the mutation insights above if they seem relevant." if random_thoughts and len(random_thoughts) > 0 else ""} Your mutation should:
1. Preserve good aspects of the current solution
2. Introduce meaningful variations
3. Aim to improve the score

MAKE SURE THE PROPOSED SOLUTION FOLLOWS THE REQUIRED FORMAT.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
DO NOT USE JSON FORMAT. USE THE SIMPLE KEY-VALUE FORMAT SHOWN BELOW.

Answer using the following schema:

name: A shortened descriptor of the mutation. Lowercase, no spaces, underscores allowed.
solution: The mutated solution as a string.
thought: The rationale for this mutation.
"""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "crossover":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join(
                    [f"- {thought}" for thought in random_thoughts]
                )
                thoughts_section = f"""

Crossover insights (consider if relevant):
{thoughts_list}

"""

            parent1 = selected_individuals[0]
            parent2 = selected_individuals[1]

            prompt = f"""
{task_description}

Current best solution (Score: {current_best_sol.evaluation_res.score:.5f}):
{current_best_sol.sol_string}

Parent 1 (Score: {parent1.evaluation_res.score:.5f}):
{parent1.sol_string}

Parent 2 (Score: {parent2.evaluation_res.score:.5f}):
{parent2.sol_string}{thoughts_section}

Create a new solution by combining elements from both parents. {"Use the crossover insights above if they seem relevant." if random_thoughts and len(random_thoughts) > 0 else ""} Your crossover should:
1. Identify and combine the best features from both parents
2. Create a coherent solution that is better than either parent
3. Explain which elements you took from each parent and why

MAKE SURE THE PROPOSED SOLUTION FOLLOWS THE REQUIRED FORMAT.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
DO NOT USE JSON FORMAT. USE THE SIMPLE KEY-VALUE FORMAT SHOWN BELOW.

Answer using the following schema:

name: A shortened descriptor of the crossover. Lowercase, no spaces, underscores allowed.
solution: The new solution combining both parents as a string.
thought: The rationale for this crossover, explaining which elements came from which parent.
"""
            return [{"role": "user", "content": prompt}]

        return []

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response for string optimization tasks

        Expected format:
        name: descriptive_name
        solution: the solution string
        thought: reasoning
        """
        if not response_str or not response_str.strip():
            return Solution("")

        content = response_str.strip()

        # Strategy 1: JSON format (try first as it has strict syntax)
        # This handles cases where LLM returns JSON despite instructions
        result = self._parse_json_format(content)
        if result and result[1]:
            cleaned_solution = self._clean_solution_string(result[1])
            return Solution(
                cleaned_solution, other_info={"name": result[0], "thought": result[2]}
            )

        # Strategy 2: Standard format parsing (expected format)
        result = self._parse_standard_format(content)
        if result and result[1]:  # Ensure we have solution
            # Clean up the solution string
            cleaned_solution = self._clean_solution_string(result[1])
            return Solution(
                cleaned_solution, other_info={"name": result[0], "thought": result[2]}
            )

        # Strategy 3: Flexible format parsing (lenient fallback)
        result = self._parse_flexible_format(content)
        if result and result[1]:
            cleaned_solution = self._clean_solution_string(result[1])
            return Solution(
                cleaned_solution, other_info={"name": result[0], "thought": result[2]}
            )

        # Strategy 4: Raw content (last resort)
        return Solution(
            content, other_info={"name": "raw", "thought": "Failed to parse"}
        )

    def _parse_standard_format(self, content: str) -> tuple:
        """Parse standard format: name -> solution -> thought order"""
        # Extract name (independent pattern)
        name_pattern = r"^name:\s*([^\n\r]+?)(?:\n|\r|$)"
        name_match = re.search(name_pattern, content, re.MULTILINE | re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else ""

        # Extract solution (capture until 'thought:' or end)
        solution_pattern = r"solution:\s*(.*?)(?=\nthought:|\Z)"
        solution_match = re.search(solution_pattern, content, re.DOTALL | re.IGNORECASE)
        solution = solution_match.group(1).strip() if solution_match else ""

        # Extract thought (independent pattern)
        thought_pattern = r"thought:\s*(.*?)$"
        thought_match = re.search(thought_pattern, content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        return (name, solution, thought)

    def _parse_flexible_format(self, content: str) -> tuple:
        """More flexible parsing for variations in format"""
        # Try to extract name anywhere in the text
        name_pattern = r"(?:name|Name|NAME)\s*:?\s*([^\n\r]+)"
        name_match = re.search(name_pattern, content, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else ""

        # Try to extract solution
        solution_pattern = r"(?:solution|Solution|SOLUTION)\s*:?\s*(.*?)(?=\n(?:thought|Thought|THOUGHT)|$)"
        solution_match = re.search(solution_pattern, content, re.DOTALL | re.IGNORECASE)
        solution = solution_match.group(1).strip() if solution_match else ""

        # Try to extract thought
        thought_pattern = r"(?:thought|Thought|THOUGHT)\s*:?\s*(.*?)$"
        thought_match = re.search(thought_pattern, content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        return (name, solution, thought)

    def _parse_json_format(self, content: str) -> tuple:
        """Parse JSON format response (fallback when LLM ignores format instructions)

        Handles cases where LLM returns JSON like:
        {
          "name": "example",
          "solution": "text",
          "thought": "reasoning"
        }
        """
        name, solution, thought = "", "", ""

        # Try to extract JSON object from content
        # First, try direct JSON parsing
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                name = str(data.get("name", ""))
                # Try different key names for solution
                solution = str(
                    data.get("solution", data.get("code", data.get("sol_string", "")))
                )
                thought = str(data.get("thought", data.get("reasoning", "")))
                return (name, solution, thought)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object within the content
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        json_matches = re.finditer(json_pattern, content, re.DOTALL)

        for match in json_matches:
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                if isinstance(data, dict):
                    name = str(data.get("name", ""))
                    solution = str(
                        data.get(
                            "solution", data.get("code", data.get("sol_string", ""))
                        )
                    )
                    thought = str(data.get("thought", data.get("reasoning", "")))

                    # If we found valid solution, return it
                    if solution:
                        return (name, solution, thought)
            except json.JSONDecodeError:
                continue

        # Try to extract from markdown code blocks containing JSON
        code_block_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        code_matches = re.finditer(
            code_block_pattern, content, re.DOTALL | re.IGNORECASE
        )

        for match in code_matches:
            try:
                json_str = match.group(1).strip()
                data = json.loads(json_str)
                if isinstance(data, dict):
                    name = str(data.get("name", ""))
                    solution = str(
                        data.get(
                            "solution", data.get("code", data.get("sol_string", ""))
                        )
                    )
                    thought = str(data.get("thought", data.get("reasoning", "")))

                    if solution:
                        return (name, solution, thought)
            except json.JSONDecodeError:
                continue

        return (name, solution, thought)

    def _clean_solution_string(self, solution: str) -> str:
        """Clean up solution string by removing wrapping quotes and escape sequences

        Args:
            solution: Raw solution string from parsing

        Returns:
            Cleaned solution string
        """
        if not solution:
            return solution

        cleaned = solution.strip()

        # Remove wrapping quotes (both single and double)
        # Handle cases like: "\"text\"" -> "text" or '"text"' -> 'text'
        if len(cleaned) >= 2:
            # Check for outer quotes
            if (cleaned[0] == '"' and cleaned[-1] == '"') or (
                cleaned[0] == "'" and cleaned[-1] == "'"
            ):
                cleaned = cleaned[1:-1]

            # After removing outer quotes, check for escaped quotes
            # "\"text\"" -> "text"
            if len(cleaned) >= 2:
                if (
                    cleaned[0] == '"'
                    and cleaned[-1] == '"'
                    and not (len(cleaned) > 2 and cleaned[1] == '"')
                ):
                    # Not already unescaped
                    pass
                elif cleaned.startswith('\\"') and cleaned.endswith('\\"'):
                    cleaned = cleaned[2:-2]
                elif cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]

        # Unescape common escape sequences
        cleaned = cleaned.replace("\\n", "\n")
        cleaned = cleaned.replace("\\t", "\t")
        cleaned = cleaned.replace('\\"', '"')
        cleaned = cleaned.replace("\\'", "'")
        cleaned = cleaned.replace("\\\\", "\\")

        return cleaned.strip()
