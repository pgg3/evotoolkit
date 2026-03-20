# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import re
from typing import List

from evotoolkit.core import Solution
from evotoolkit.evo_method.evoengineer import EvoEngineerInterface, Operator
from evotoolkit.task.python_task.python_task import PythonTask


class EvoEngineerPythonInterface(EvoEngineerInterface):
    """EvoEngineer Adapter for Python code optimization tasks.

    This class provides EvoEngineer algorithm logic for Python tasks.
    Task behavior is driven by the bound task specification.
    """

    def __init__(self, task: PythonTask):
        super().__init__(task)

    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for Python optimization"""
        return [Operator("init", 0)]

    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for Python optimization"""
        return [Operator("crossover", 2), Operator("mutation", 1)]

    def get_operator_prompt(
        self,
        operator_name: str,
        selected_individuals: List[Solution],
        current_best_sol: Solution,
        random_descriptions: List[str],
        **kwargs,
    ) -> List[dict]:
        """Generate prompt for any operator"""
        task_description = self.task.spec.prompt

        if current_best_sol is None:
            current_best_sol = self._make_initial_solution()

        current_best_score = self._format_solution_score(current_best_sol)

        if operator_name == "init":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_descriptions:
                thoughts_list = "\n".join([f"- {thought}" for thought in random_descriptions])
                thoughts_section = f"""{thoughts_list}"""

            prompt = f"""# PYTHON FUNCTION OPTIMIZATION TASK
{task_description}

## INITIAL SOLUTION
**Name:** {current_best_sol.metadata.name or "Initial"}
**Score:** {current_best_score}
**Current Approach:** {current_best_sol.metadata.description or "Initial"}
**Function Code:**
```python
{current_best_sol.sol_string}
```

## OPTIMIZATION INSIGHTS
{thoughts_section}

## OPTIMIZATION STRATEGY
{"Use the insights above if relevant as optimization guidance." if random_descriptions else ""}
Propose a new Python function that aims to improve the score while ensuring it returns the correct result.

## RESPONSE FORMAT:
name: [descriptive_name_with_underscores]
code:
```python
[Your Python implementation]
```
thought: [The rationale for the improvement idea.]

## FORMAT REQUIREMENTS:
1. The code MUST be wrapped in ```python and ``` markers
2. MAKE SURE THE PROPOSAL CODE IS VALID PYTHON CODE."""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "crossover":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_descriptions:
                thoughts_list = "\n".join([f"- {thought}" for thought in random_descriptions])
                thoughts_section = f"""{thoughts_list}"""

            # Build parent functions info
            parents_info = ""
            for i, parent in enumerate(selected_individuals, 1):
                parents_info += f"""
**Parent {i}:**
**Name:** {parent.metadata.name or f"function_{i}"}
**Score:** {self._format_solution_score(parent)}
**Parent Approach:** {parent.metadata.description or "No thought provided"}
**Function Code:**
```python
{parent.sol_string}
```
"""

            prompt = f"""# PYTHON FUNCTION CROSSOVER TASK
{task_description}

## CURRENT BEST
**Name:** {current_best_sol.metadata.name or "current_best"}
**Score:** {current_best_score}
**Current Approach:** {current_best_sol.metadata.description or "Current best implementation"}
**Function Code:**
```python
{current_best_sol.sol_string}
```

## PARENTS TO COMBINE
{parents_info}

## OPTIMIZATION INSIGHTS
{thoughts_section}

## CROSSOVER STRATEGY
Combine the best features from both parent functions:
{"Use the insights above if relevant as crossover guidance." if random_descriptions else ""}

Create a hybrid Python function that combines the strengths of both parents.

## RESPONSE FORMAT:
name: [descriptive_name_with_underscores]
code:
```python
[Your Python implementation]
```
thought: [The rationale for the improvement idea.]

## FORMAT REQUIREMENTS:
1. The code MUST be wrapped in ```python and ``` markers
2. MAKE SURE THE PROPOSAL CODE IS VALID PYTHON CODE."""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "mutation":
            individual = selected_individuals[0]

            # Build the thoughts section if available
            thoughts_section = ""
            if random_descriptions:
                thoughts_list = "\n".join([f"- {thought}" for thought in random_descriptions])
                thoughts_section = f"""{thoughts_list}"""

            prompt = f"""# PYTHON FUNCTION MUTATION TASK
{task_description}

## CURRENT BEST
**Name:** {current_best_sol.metadata.name or "current_best"}
**Score:** {current_best_score}
**Previous Approach:** {current_best_sol.metadata.description or "Current best implementation"}
**Function Code:**
```python
{current_best_sol.sol_string}
```

## SOURCE TO MUTATE
**Name:** {individual.metadata.name or "mutation_base"}
**Score:** {self._format_solution_score(individual)}
**Target Approach:** {individual.metadata.description or "No thought provided"}
**Function Code:**
```python
{individual.sol_string}
```

## OPTIMIZATION INSIGHTS
{thoughts_section}

## MUTATION STRATEGY
Apply significant changes to the target function:
{"Use the insights above if relevant as mutation guidance." if random_descriptions else ""}
Create a substantially modified version that explores new optimization directions.

## RESPONSE FORMAT:
name: [descriptive_name_with_underscores]
code:
```python
[Your Python implementation]
```
thought: [The rationale for the improvement idea.]

## FORMAT REQUIREMENTS:
1. The code MUST be wrapped in ```python and ``` markers
2. MAKE SURE THE PROPOSAL CODE IS VALID PYTHON CODE."""
            return [{"role": "user", "content": prompt}]
        else:
            raise ValueError(f"Unknown operator: {operator_name}")

    def parse_response(self, response_str: str) -> Solution:
        """Improved parser with multiple fallback strategies"""
        if not response_str or not response_str.strip():
            return Solution("")

        content = response_str.strip()

        # Strategy 1: Standard format parsing (most reliable)
        result = self._parse_standard_format(content)
        if result and result[1]:  # Ensure we have code
            return self.make_solution(result[1], name=result[0], description=result[2])

        # Strategy 2: Flexible format parsing
        result = self._parse_flexible_format(content)
        if result and result[1]:
            return self.make_solution(result[1], name=result[0], description=result[2])

        # Strategy 3: Code block fallback
        code = self._extract_any_code_block(content)
        if code:
            return self.make_solution(code, name="extracted", description="Fallback parsing")

        # Strategy 4: Raw content (last resort)
        return self.make_solution(content, name="raw", description="Failed to parse")

    def _parse_standard_format(self, content: str) -> tuple:
        """Parse standard format: name -> code -> thought order"""
        # Extract name (independent pattern)
        name_pattern = r"^name:\s*([^\n\r]+?)(?:\n|\r|$)"
        name_match = re.search(name_pattern, content, re.MULTILINE | re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else ""

        # Extract code block (independent pattern)
        code_pattern = r"code:\s*\n*```(?:python|py)?\\n(.*?)```"
        code_match = re.search(code_pattern, content, re.DOTALL | re.IGNORECASE)
        code = code_match.group(1).strip() if code_match else ""

        # Extract thought (independent pattern)
        thought_pattern = r"thought:\s*(.*?)$"
        thought_match = re.search(thought_pattern, content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        return (name, code, thought)

    def _parse_flexible_format(self, content: str) -> tuple:
        """More flexible parsing for variations in format"""
        # Try to extract name anywhere in the text
        name_pattern = r"(?:name|Name|NAME)\s*:?\s*([^\n\r]+)"
        name_match = re.search(name_pattern, content, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else ""

        # Try to extract any code block
        code = self._extract_any_code_block(content)

        # Try to extract thought
        thought_pattern = r"(?:thought|Thought|THOUGHT)\s*:?\s*(.*?)(?=\n(?:name|code)|$)"
        thought_match = re.search(thought_pattern, content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        return (name, code, thought)

    def _extract_any_code_block(self, content: str) -> str:
        """Extract any code block from the content"""
        # Priority 1: Look for ```python or ```py blocks
        python_pattern = r"```(?:python|py)\n(.*?)```"
        match = re.search(python_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Priority 2: Look for any ``` blocks
        generic_pattern = r"```[^\n]*\n(.*?)```"
        match = re.search(generic_pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Priority 3: Look for code: section without proper markers
        code_pattern = r"code:\s*\n*(.*?)(?=\n(?:thought|$))"
        match = re.search(code_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            code_content = match.group(1).strip()
            # Remove any remaining ``` markers
            code_content = re.sub(r"^```[^\n]*\n?", "", code_content)
            code_content = re.sub(r"\n?```\s*$", "", code_content)
            return code_content.strip()

        return ""
