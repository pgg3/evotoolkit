# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""EoH Interface for CANN Init Task

This module implements the Evolution of Heuristics (EoH) interface for CANN operator optimization.
It generates prompts for LLM to produce 6 components for Ascend C operators.

EoH Operators:
- I1: Initialization - Generate from scratch
- E1: Crossover - Combine features from multiple individuals
- E2: Guided Crossover - Combine based on common backbone
- M1: Mutation - Modified version of an individual
- M2: Parameter Mutation - Change parameters/configuration
"""

import json
import re
from typing import Any, Dict, List, Optional

from evotoolkit.core import EoHInterface, Solution

from ..cann_init_task import CANNInitTask


class EoHCANNInterface(EoHInterface):
    """Evolution of Heuristics interface for CANN kernel optimization.

    Generates prompts for LLM to produce 6 components:
    1. kernel_impl - Kernel class and helper code
    2. kernel_entry_body - Entry function body
    3. tiling_fields - TilingData struct fields
    4. tiling_func_body - TilingFunc function body
    5. infer_shape_body - InferShape function body
    6. output_alloc_code - Output tensor allocation code
    """

    def __init__(self, task: CANNInitTask):
        super().__init__(task)
        self.task: CANNInitTask = task

    def get_prompt_i1(self) -> List[dict]:
        """Generate initialization prompt (I1 operator)."""
        task_description = self.task.get_task_description()

        prompt = f"""{task_description}

## Task

Generate an optimized Ascend C operator implementation. Focus on:
- Correct functionality matching the Python reference
- Efficient memory access patterns
- Proper use of Ascend C APIs (TPipe, TQue, DataCopy, etc.)
- Appropriate tiling strategy for the target hardware

{self._build_output_format()}"""

        return [{"role": "user", "content": prompt}]

    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E1 (crossover) prompt."""
        task_description = self.task.get_task_description()

        # Build individuals section
        individuals_section = self._build_individuals_section(selected_individuals)

        prompt = f"""{task_description}

## Existing Implementations

I have {len(selected_individuals)} existing implementations:

{individuals_section}

## Task

Create a new implementation that has a totally different form from the given ones.
Combine diverse optimization strategies from the existing implementations.

{self._build_output_format()}"""

        return [{"role": "user", "content": prompt}]

    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E2 (guided crossover) prompt."""
        task_description = self.task.get_task_description()

        # Build individuals section
        individuals_section = self._build_individuals_section(selected_individuals)

        prompt = f"""{task_description}

## Existing Implementations

{individuals_section}

## Task

1. First, identify the common backbone idea in the provided implementations
2. Based on the backbone, create a new implementation that:
   - Inherits the successful patterns
   - Adds novel optimizations
   - Maintains correctness

{self._build_output_format()}"""

        return [{"role": "user", "content": prompt}]

    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """Generate M1 (mutation) prompt."""
        task_description = self.task.get_task_description()

        # Format the individual's components
        individual_section = self._format_solution(individual)

        prompt = f"""{task_description}

## Current Implementation

{individual_section}

## Task

Create a modified version of the implementation that:
- Has a different algorithmic approach
- Explores alternative optimization directions
- Maintains correctness while potentially improving performance

{self._build_output_format()}"""

        return [{"role": "user", "content": prompt}]

    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        """Generate M2 (parameter mutation) prompt."""
        task_description = self.task.get_task_description()

        # Format the individual's components
        individual_section = self._format_solution(individual)

        prompt = f"""{task_description}

## Current Implementation

{individual_section}

## Task

Identify the main parameters (tile size, buffer count, block dimension, etc.) and create a new implementation with:
- Different parameter settings
- Optimized configuration for the hardware
- Same algorithmic structure but tuned values

{self._build_output_format()}"""

        return [{"role": "user", "content": prompt}]

    def _build_individuals_section(self, individuals: List[Solution]) -> str:
        """Build section showing multiple individuals."""
        parts = []
        for i, ind in enumerate(individuals, 1):
            info = ind.other_info or {}
            name = info.get("name", f"implementation_{i}")
            thought = info.get("thought", info.get("algorithm", ""))

            parts.append(f"### Implementation {i}: {name}")
            if thought:
                parts.append(f"**Approach:** {thought}")
            parts.append("")
            parts.append(self._format_solution_components(ind))
            parts.append("")

        return "\n".join(parts)

    def _format_solution(self, solution: Solution) -> str:
        """Format a single solution for display."""
        info = solution.other_info or {}
        name = info.get("name", "unnamed")
        thought = info.get("thought", info.get("algorithm", ""))

        parts = [f"**Name:** {name}"]
        if thought:
            parts.append(f"**Approach:** {thought}")
        parts.append("")
        parts.append(self._format_solution_components(solution))

        return "\n".join(parts)

    def _format_solution_components(self, solution: Solution) -> str:
        """Format solution's 6 components."""
        return self.task.format_solution_components(solution)

    def _build_output_format(self) -> str:
        """Build output format specification."""
        return """---

## Output Format

Provide a name and approach description, then all 6 required components:

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

IMPORTANT:
- All code must be valid Ascend C code
- Follow the exact format with ### headers and code blocks
- Provide clear name and thought explaining your approach"""

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response to extract name, thought, and 6 components."""
        if not response_str or not response_str.strip():
            return Solution("")

        # Extract name
        name = self._extract_name(response_str)

        # Extract thought (also try "algorithm" for backwards compatibility)
        thought = self._extract_thought(response_str)

        # Extract 6 components using the same logic as FunSearch
        components = self._extract_components(response_str)

        # Add name and thought to components
        components["name"] = name or "eoh_generated"
        components["thought"] = thought or ""

        return Solution(sol_string="", other_info=components)

    def _extract_name(self, response_str: str) -> str:
        """Extract name from response."""
        # Try standard format
        match = re.search(r"^name:\s*([^\n]+)", response_str, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_thought(self, response_str: str) -> str:
        """Extract thought/algorithm description from response."""
        # Try "thought:" format
        match = re.search(r"^thought:\s*([^\n]+)", response_str, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try boxed format {description} for backwards compatibility
        match = re.search(r"\{([^}]+)\}", response_str)
        if match:
            return match.group(1).strip()

        return ""

    def _extract_components(self, response_str: str) -> Dict[str, Any]:
        """Extract components from structured response.

        Expected format:
        ### SECTION_NAME
        ```lang
        content
        ```
        """
        components: Dict[str, Any] = {}

        # Pattern to match ### SECTION_NAME followed by code block
        pattern = r"###\s*(\w+).*?\n```(?:\w+)?\s*\n(.*?)\n```"

        for match in re.finditer(pattern, response_str, re.DOTALL | re.IGNORECASE):
            section_name = match.group(1).upper()
            content = match.group(2).strip()

            # Map section names to component keys
            section_map = {
                "KERNEL_IMPL": "kernel_impl",
                "KERNEL_ENTRY_BODY": "kernel_entry_body",
                "TILING_FIELDS": "tiling_fields",
                "TILING_FUNC_BODY": "tiling_func_body",
                "INFER_SHAPE_BODY": "infer_shape_body",
                "OUTPUT_ALLOC_CODE": "output_alloc_code",
                "KERNEL_INCLUDES": "kernel_includes",
                "TILING_INCLUDES": "tiling_includes",
                "TILING_FUNC_INCLUDES": "tiling_func_includes",
            }

            if section_name in section_map:
                key = section_map[section_name]

                # Special handling for tiling_fields
                if key == "tiling_fields":
                    components[key] = self._parse_tiling_fields(content)
                # Special handling for include lists
                elif key in ("kernel_includes", "tiling_includes", "tiling_func_includes"):
                    components[key] = self._parse_includes(content)
                else:
                    components[key] = content

        return components

    def _parse_tiling_fields(self, content: str) -> List[Dict[str, Any]]:
        """Parse tiling fields from text format.

        Supported formats:
        - TYPE NAME
        - TYPE NAME[SIZE]
        - struct TYPE NAME
        """
        content = content.strip()

        # Try JSON first (backwards compatibility)
        if content.startswith("["):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        # Parse text format
        fields = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue

            field = self._parse_tiling_field_line(line)
            if field:
                fields.append(field)

        return fields

    def _parse_tiling_field_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single tiling field line."""
        # Pattern for struct: struct TYPE NAME
        struct_match = re.match(r"struct\s+(\w+)\s+(\w+)$", line, re.IGNORECASE)
        if struct_match:
            return {
                "name": struct_match.group(2),
                "type": struct_match.group(1),
                "is_struct": True,
            }

        # Pattern for array: TYPE NAME[SIZE]
        array_match = re.match(r"(\w+)\s+(\w+)\[(\d+)\]", line)
        if array_match:
            return {
                "name": array_match.group(2),
                "type": array_match.group(1),
                "size": int(array_match.group(3)),
            }

        # Pattern for scalar: TYPE NAME
        scalar_match = re.match(r"(\w+)\s+(\w+)$", line)
        if scalar_match:
            return {
                "name": scalar_match.group(2),
                "type": scalar_match.group(1),
            }

        return None

    def _parse_includes(self, content: str) -> List[str]:
        """Parse include list from text format."""
        content = content.strip()

        # Filter markdown artifacts
        if content in ('```', '```cpp', '```c++', '```c', '```h'):
            return []

        # Try JSON first
        if content.startswith("["):
            try:
                parsed = json.loads(content)
                return [h for h in parsed if self._is_valid_header_path(h)]
            except json.JSONDecodeError:
                pass

        # Parse as text (one per line)
        includes = []
        for line in content.split("\n"):
            line = line.strip().strip('"').strip("'")
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            if self._is_valid_header_path(line):
                includes.append(line)
        return includes

    def _is_valid_header_path(self, path: str) -> bool:
        """Check if a string looks like a valid C/C++ header path."""
        if not path or not path.strip():
            return False
        if path.startswith('`') or path.endswith('`'):
            return False
        if '.h' in path or '/' in path:
            return True
        if any(path.endswith(ext) for ext in ('.hpp', '.hxx', '.h++')):
            return True
        return False
