# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Prompt generation for EvoEngineer CANN Interface"""

import json
from typing import List

from evotoolkit.core import Solution


class PromptMixin:
    """Mixin for prompt generation methods"""

    def _get_task_description(self) -> str:
        """Get task description from CANNInitTask"""
        return self.task.get_base_task_description()

    def _format_solution(self, sol: Solution) -> str:
        """Format a solution for display in prompt"""
        if not sol or not sol.sol_string:
            return "(no solution)"

        info = sol.other_info or {}
        runtime = ""
        if sol.evaluation_res and sol.evaluation_res.score is not None:
            runtime = f"Runtime: {-sol.evaluation_res.score:.4f} ms\n"

        return f"""Name: {info.get('name', 'unnamed')}
{runtime}Thought: {info.get('thought', '')}

kernel_src:
```cpp
{sol.sol_string}
```

tiling_fields:
```json
{json.dumps(info.get('tiling_fields', []), indent=2)}
```

tiling_func_body:
```cpp
{info.get('tiling_func_body', '')}
```

infer_shape_body:
```cpp
{info.get('infer_shape_body', '')}
```"""

    def _get_response_format(self) -> str:
        return """## RESPONSE FORMAT
name: [descriptive_name]
thought: [optimization rationale]

kernel_src:
```cpp
[Ascend C kernel code]
```

tiling_fields:
```json
[{"name": "field1", "type": "uint32_t"}, ...]
```

tiling_func_body:
```cpp
[TilingFunc body]
```

infer_shape_body:
```cpp
[InferShape body]
```"""

    def _get_init_prompt(self, task_desc: str, current_best: Solution, thoughts: List[str]) -> List[dict]:
        thoughts_section = ""
        if thoughts:
            thoughts_section = "\n## OPTIMIZATION INSIGHTS\n" + "\n".join(f"- {t}" for t in thoughts)

        best_section = ""
        if current_best and current_best.sol_string:
            best_section = f"\n## CURRENT BEST\n{self._format_solution(current_best)}"

        prompt = f"""# ASCEND C KERNEL TASK

{task_desc}
{best_section}
{thoughts_section}

## TASK
Implement a high-performance Ascend C kernel for this operator.

{self._get_response_format()}"""

        return [{"role": "user", "content": prompt}]

    def _get_crossover_prompt(
        self, task_desc: str, parents: List[Solution], current_best: Solution, thoughts: List[str]
    ) -> List[dict]:
        thoughts_section = ""
        if thoughts:
            thoughts_section = "\n## OPTIMIZATION INSIGHTS\n" + "\n".join(f"- {t}" for t in thoughts)

        parents_section = "\n## PARENTS TO COMBINE\n"
        for i, p in enumerate(parents, 1):
            parents_section += f"\n### Parent {i}\n{self._format_solution(p)}\n"

        prompt = f"""# ASCEND C KERNEL CROSSOVER

{task_desc}

## CURRENT BEST
{self._format_solution(current_best)}
{parents_section}
{thoughts_section}

## TASK
Combine the best features from both parent implementations.

{self._get_response_format()}"""

        return [{"role": "user", "content": prompt}]

    def _get_mutation_prompt(
        self, task_desc: str, individual: Solution, current_best: Solution, thoughts: List[str]
    ) -> List[dict]:
        thoughts_section = ""
        if thoughts:
            thoughts_section = "\n## OPTIMIZATION INSIGHTS\n" + "\n".join(f"- {t}" for t in thoughts)

        prompt = f"""# ASCEND C KERNEL MUTATION

{task_desc}

## CURRENT BEST
{self._format_solution(current_best)}

## SOURCE TO MUTATE
{self._format_solution(individual)}
{thoughts_section}

## TASK
Apply significant modifications to explore new optimization directions.

{self._get_response_format()}"""

        return [{"role": "user", "content": prompt}]
