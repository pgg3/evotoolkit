# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Prompt generation for EvoEngineer CANN Interface

This module generates prompts that require LLM to produce 6 components for Ascend C operators,
along with performance profiling information and optimization insights.
"""

from typing import List

from evotoolkit.core import Solution


class PromptMixin:
    """Mixin for prompt generation methods"""

    def _get_task_description(self) -> str:
        """Get full task description from CANNInitTask"""
        return self.task.get_task_description()

    def _format_solution(self, sol: Solution, show_profile: bool = True) -> str:
        """Format a solution for display in prompt.

        Args:
            sol: Solution to format
            show_profile: Whether to show performance profile
        """
        if not sol or not sol.other_info:
            return "(no solution)"

        info = sol.other_info

        parts = []

        # Name and thought
        name = info.get("name", "unnamed")
        thought = info.get("thought", "")
        parts.append(f"**Name:** {name}")

        # Runtime if available
        if sol.evaluation_res and sol.evaluation_res.score is not None:
            runtime = -sol.evaluation_res.score
            parts.append(f"**Runtime:** {runtime:.5f} milliseconds")

        if thought:
            parts.append(f"**Approach:** {thought}")

        parts.append("")

        # 6 components
        parts.append(self.task.format_solution_components(sol))

        # Performance profile if available
        if show_profile and sol.evaluation_res and sol.evaluation_res.additional_info:
            prof_string = sol.evaluation_res.additional_info.get("prof_string", "")
            if prof_string:
                parts.append("")
                parts.append("**Performance Profile:**")
                parts.append(prof_string)

        return "\n".join(parts)

    def _build_thoughts_section(self, thoughts: List[str]) -> str:
        """Build optimization insights section from random thoughts."""
        if not thoughts:
            return ""

        thoughts_list = "\n".join(f"- {t}" for t in thoughts)
        return f"""
## OPTIMIZATION INSIGHTS

{thoughts_list}
"""

    def _build_output_format(self) -> str:
        """Build output format specification."""
        return """## RESPONSE FORMAT

name: [descriptive_name_with_underscores]
thought: [Your optimization strategy and rationale]

### KERNEL_IMPL
```cpp
[Your kernel class and helper code]
```

### KERNEL_ENTRY_BODY
```cpp
[Your entry function body]
```

### TILING_FIELDS
```
[Your tiling fields, one per line]
```

Format: `TYPE NAME` or `TYPE NAME[SIZE]` or `struct TYPE NAME`

Example:
```
uint32_t totalLength
uint32_t tileNum
int64_t dims[4]
struct TCubeTiling matmulTiling
```

### TILING_FUNC_BODY
```cpp
[Your tiling function body]
```

### INFER_SHAPE_BODY
```cpp
[Your infer shape body]
```

### OUTPUT_ALLOC_CODE
```cpp
[Your output allocation code]
```

### TILING_INCLUDES (optional, only if using struct types like TCubeTiling)
```
tiling/platform/platform_ascendc.h
```

### KERNEL_INCLUDES (optional, only if needed)
```
lib/matmul_intf.h
```

## FORMAT REQUIREMENTS
1. All code must be valid Ascend C code
2. Follow the exact format with ### headers and code blocks
3. Provide clear name and thought explaining your approach
4. MAKE SURE THE PROPOSAL CODE IS VALID ASCEND C CODE"""

    def _get_init_prompt(
        self,
        task_desc: str,
        thoughts: List[str],
    ) -> List[dict]:
        """Generate init operator prompt."""
        thoughts_section = self._build_thoughts_section(thoughts)

        strategy_section = """## OPTIMIZATION STRATEGY

"""
        if thoughts:
            strategy_section += "Use the insights above if relevant as optimization guidance.\n"
        strategy_section += "Propose a new Ascend C kernel implementation which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result."

        prompt = f"""# ASCEND C KERNEL OPTIMIZATION TASK

{task_desc}
{thoughts_section}
{strategy_section}

---

{self._build_output_format()}"""

        return [{"role": "user", "content": prompt}]

    def _get_crossover_prompt(
        self,
        task_desc: str,
        parents: List[Solution],
        thoughts: List[str],
    ) -> List[dict]:
        """Generate crossover operator prompt."""
        thoughts_section = self._build_thoughts_section(thoughts)

        # Build parents section
        parents_section = "\n## PARENTS TO COMBINE\n"
        for i, p in enumerate(parents, 1):
            parents_section += f"\n### Parent {i}\n\n{self._format_solution(p, show_profile=False)}\n"

        strategy_section = """## CROSSOVER STRATEGY

Combine the best features from both parent kernels:
"""
        if thoughts:
            strategy_section += "Use the insights above if relevant as crossover guidance.\n"
        strategy_section += """
Create a hybrid Ascend C kernel that combines the strengths of both parents."""

        prompt = f"""# ASCEND C KERNEL CROSSOVER TASK

{task_desc}
{parents_section}

{thoughts_section}
{strategy_section}

---

{self._build_output_format()}"""

        return [{"role": "user", "content": prompt}]

    def _get_mutation_prompt(
        self,
        task_desc: str,
        individual: Solution,
        thoughts: List[str],
    ) -> List[dict]:
        """Generate mutation operator prompt."""
        thoughts_section = self._build_thoughts_section(thoughts)

        strategy_section = """## MUTATION STRATEGY

Apply significant changes to the target kernel:
"""
        if thoughts:
            strategy_section += "Use the insights above if relevant as mutation guidance.\n"
        strategy_section += "Create a substantially modified version that explores new optimization directions."

        prompt = f"""# ASCEND C KERNEL MUTATION TASK

{task_desc}

## SOURCE TO MUTATE

{self._format_solution(individual, show_profile=False)}

{thoughts_section}
{strategy_section}

---

{self._build_output_format()}"""

        return [{"role": "user", "content": prompt}]
