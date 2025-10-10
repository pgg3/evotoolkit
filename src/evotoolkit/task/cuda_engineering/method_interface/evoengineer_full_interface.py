# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import re
from typing import List

from evotoolkit.core import EvoEngineerInterface, Operator, Solution

from ..cuda_task import CudaTask


class EvoEngineerFullCudaInterface(EvoEngineerInterface):
    def __init__(self, task: CudaTask):
        super().__init__(task)

    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for CUDA optimization"""
        return [Operator("init", 0)]

    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for CUDA optimization"""
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

        if current_best_sol is None:
            current_best_sol = self.make_init_sol()

        if operator_name == "init":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join(
                    [f"- {thought}" for thought in random_thoughts]
                )
                thoughts_section = f"""
{thoughts_list}
"""

            prompt = f"""# CUDA KERNEL OPTIMIZATION TASK
{task_description}

## BASELINE CODE
**Name:** {current_best_sol.other_info["name"]}
**Runtime:** {-current_best_sol.evaluation_res.score:.5f} milliseconds
**Current Approach:** {current_best_sol.other_info["thought"]}
**Kernel Code:**
```cpp
{current_best_sol.sol_string}
```
**Performance Profile:**
{current_best_sol.evaluation_res.additional_info["prof_string"]}

## OPTIMIZATION INSIGHTS
{thoughts_section}

## OPTIMIZATION STRATEGY
{"Use the insights above if relevant as optimization guidance." if random_thoughts and len(random_thoughts) > 0 else ""}
Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.

## RESPONSE FORMAT:
name: [descriptive_name_with_underscores]
code:
```cpp
[Your CUDA kernel implementation]
```
thought: [The rationale for the improvement idea.]

## FORMAT REQUIREMENTS:
1. The code MUST be wrapped in ```cpp and ``` markers
2. The PYBIND11_MODULE inside the code has to be the same as in the ## BASELINE CODE.
3. MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE."""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "crossover":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join(
                    [f"- {thought}" for thought in random_thoughts]
                )
                thoughts_section = f"""
{thoughts_list}
"""
            # Build parent kernels info
            parents_info = ""
            for i, parent in enumerate(selected_individuals, 1):
                parents_info += f"""
**Parent {i}:**
**Name:** {parent.other_info["name"]} 
**Runtime:** {-parent.evaluation_res.score:.5f} milliseconds
**Parent Approach:** {parent.other_info["thought"]}
**Kernel Code:**
```cpp
{parent.sol_string}
```
"""
            prompt = f"""# CUDA KERNEL CROSSOVER TASK
{task_description}

## BASELINE CODE
**Name:** {current_best_sol.other_info["name"]}
**Runtime:** {-current_best_sol.evaluation_res.score:.5f} milliseconds
**Current Approach:** {current_best_sol.other_info["thought"]}
**Kernel Code:**
```cpp
{current_best_sol.sol_string}
```
**Performance Profile:**
{current_best_sol.evaluation_res.additional_info["prof_string"]}

## PARENTS TO COMBINE

{parents_info}

## OPTIMIZATION INSIGHTS
{thoughts_section}

## CROSSOVER STRATEGY
Combine the best features from both parent kernels:
{"Use the insights above if relevant as crossover guidance." if random_thoughts and len(random_thoughts) > 0 else ""}

Create a hybrid CUDA kernel that combines the strengths of both parents.

## RESPONSE FORMAT:
name: [descriptive_name_with_underscores]
code:
```cpp
[Your CUDA kernel implementation]
```
thought: [The rationale for the improvement idea.]

## FORMAT REQUIREMENTS:
1. The code MUST be wrapped in ```cpp and ``` markers
2. The PYBIND11_MODULE inside the code has to be the same as in the ## BASELINE CODE.
3. MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE."""
            return [{"role": "user", "content": prompt}]
        elif operator_name == "mutation":
            individual = selected_individuals[0]

            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join(
                    [f"- {thought}" for thought in random_thoughts]
                )
                thoughts_section = f"""
{thoughts_list}
"""
            prompt = f"""# CUDA KERNEL MUTATION TASK
{task_description}

## CURRENT BEST
**Name:** {current_best_sol.other_info["name"]}
**Runtime:** {-current_best_sol.evaluation_res.score:.5f} milliseconds
**Previous Approach:** {current_best_sol.other_info["thought"]}
**Kernel Code:**
```cpp
{current_best_sol.sol_string}
```
**Performance Profile:**
{current_best_sol.evaluation_res.additional_info["prof_string"]}

## SOURCE TO MUTATE
**Name:** {individual.other_info["name"]}
**Runtime:** {-individual.evaluation_res.score:.5f} milliseconds
**Target Approach:** {individual.other_info["thought"]}
**Kernel Code:**
```cpp
{individual.sol_string}
```

## OPTIMIZATION INSIGHTS
{thoughts_section}

## MUTATION STRATEGY
Apply significant changes to the target kernel:
{"Use the insights above if relevant as mutation guidance." if random_thoughts and len(random_thoughts) > 0 else ""}
Create a substantially modified version that explores new optimization directions.

## RESPONSE FORMAT:
name: [descriptive_name_with_underscores]
code:
```cpp
[Your CUDA kernel implementation]
```
thought: [The rationale for the improvement idea.]

## FORMAT REQUIREMENTS:
1. The code MUST be wrapped in ```cpp and ``` markers
2. The PYBIND11_MODULE inside the code has to be the same as in the ## BASELINE CODE.
3. MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE."""
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
            return Solution(
                result[1], other_info={"name": result[0], "thought": result[2]}
            )

        # Strategy 2: Flexible format parsing
        result = self._parse_flexible_format(content)
        if result and result[1]:
            return Solution(
                result[1], other_info={"name": result[0], "thought": result[2]}
            )

        # Strategy 3: Code block fallback
        code = self._extract_any_code_block(content)
        if code:
            return Solution(
                code, other_info={"name": "extracted", "thought": "Fallback parsing"}
            )

        # Strategy 4: Raw content (last resort)
        return Solution(
            content, other_info={"name": "raw", "thought": "Failed to parse"}
        )

    def _parse_standard_format(self, content: str) -> tuple:
        """Parse standard format: name -> code -> thought order"""
        # Extract name (independent pattern)
        name_pattern = r"^name:\s*([^\n\r]+?)(?:\n|\r|$)"
        name_match = re.search(name_pattern, content, re.MULTILINE | re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else ""

        # Extract code block (independent pattern)
        code_pattern = r"code:\s*\n*```(?:cpp|c\+\+|cuda)?\n(.*?)```"
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
        thought_pattern = (
            r"(?:thought|Thought|THOUGHT)\s*:?\s*(.*?)(?=\n(?:name|code)|$)"
        )
        thought_match = re.search(thought_pattern, content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        return (name, code, thought)

    def _extract_any_code_block(self, content: str) -> str:
        """Extract any code block from the content"""
        # Priority 1: Look for ```cpp or ```c++ blocks
        cpp_pattern = r"```(?:cpp|c\+\+|cuda)\n(.*?)```"
        match = re.search(cpp_pattern, content, re.DOTALL | re.IGNORECASE)
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
