# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""FunSearch Interface for CANN Init Task

This module implements the FunSearch method interface for CANN operator optimization.
It generates prompts that require LLM to produce 6 components for Ascend C operators.
"""

import json
import re
from typing import Any, Dict, List, Optional

from evotoolkit.core import FunSearchInterface, Solution

from ..cann_init_task import CANNInitTask


class FunSearchCANNInterface(FunSearchInterface):
    """FunSearch interface for CANN kernel optimization.

    Generates prompts for LLM to produce 6 components:
    1. kernel_impl - Kernel class and helper code
    2. kernel_entry_body - Entry function body
    3. tiling_fields - TilingData struct fields (JSON)
    4. tiling_func_body - TilingFunc function body
    5. infer_shape_body - InferShape function body
    6. output_alloc_code - Output tensor allocation code

    Optional components:
    - kernel_includes - Extra kernel headers
    - tiling_func_includes - Extra TilingFunc headers
    """

    def __init__(self, task: CANNInitTask):
        super().__init__(task)
        self.task: CANNInitTask = task  # Type hint for IDE

    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate FunSearch prompt based on available solutions.

        Args:
            solutions: List of existing solutions for reference/comparison.
                - Empty list: Generate from scratch
                - 1 solution: Use as reference to optimize
                - 2+ solutions: Show worse → better comparison

        Returns:
            List of message dicts for LLM API call.
        """
        # Get complete task description from task (includes signature + component spec)
        task_description = self.task.get_task_description()

        # Build solution section based on available solutions (FunSearch 特有)
        if len(solutions) >= 2:
            solution_section = self._build_comparison_section(solutions[0], solutions[1])
        elif len(solutions) == 1:
            solution_section = self._build_reference_section(solutions[0])
        else:
            solution_section = self._build_initial_section()

        # Build output format section (FunSearch 特有)
        output_format = self._build_output_format()

        # Assemble: task_description + solution_section + output_format
        prompt = f"""{task_description}

{solution_section}

{output_format}"""

        return [{"role": "user", "content": prompt}]

    def _build_initial_section(self) -> str:
        """Build section for initial generation (no existing solutions)."""
        return """## Task

Generate an optimized Ascend C operator implementation. Focus on:
- Correct functionality matching the Python reference
- Efficient memory access patterns
- Proper use of Ascend C APIs (TPipe, TQue, DataCopy, etc.)
- Appropriate tiling strategy for the target hardware"""

    def _build_reference_section(self, solution: Solution) -> str:
        """Build section showing a reference solution to optimize."""
        components = self.task.format_solution_components(solution)

        return f"""## Reference Implementation

Here is an existing implementation that you should optimize:

{components}

## Task

Propose an improved implementation that:
- Maintains correctness (same output as reference)
- Improves performance (reduces runtime)
- Uses better memory access patterns or tiling strategies"""

    def _build_comparison_section(self, worse: Solution, better: Solution) -> str:
        """Build section showing worse → better comparison."""
        worse_components = self.task.format_solution_components(worse)
        better_components = self.task.format_solution_components(better)

        return f"""## Implementation Comparison

### Baseline Implementation (slower)

{worse_components}

### Improved Implementation (faster)

{better_components}

## Task

Analyze the improvements made in the faster implementation and propose an even better version that:
- Maintains correctness
- Further improves performance
- Applies additional optimizations (e.g., better tiling, vectorization, memory coalescing)"""

    def _build_output_format(self) -> str:
        """Build output format specification."""
        return """---

## Output Format

Respond with all 6 required components using exactly this format:

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

IMPORTANT:
- All code must be valid Ascend C code
- Follow the exact format with ### headers and code blocks
- Do not include explanations outside the code blocks"""

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response to extract 6 components.

        Args:
            response_str: Raw LLM response text.

        Returns:
            Solution with sol_string="" and other_info containing the 6 components.
        """
        components = self._extract_components(response_str)

        # Validate required components
        required = [
            "kernel_impl",
            "kernel_entry_body",
            "tiling_fields",
            "tiling_func_body",
            "infer_shape_body",
            "output_alloc_code",
        ]

        missing = [k for k in required if not components.get(k)]
        if missing:
            # Try fallback parsing for partial responses
            components = self._fallback_parse(response_str, components)

        return Solution(sol_string="", other_info=components)

    def _extract_components(self, response_str: str) -> Dict[str, Any]:
        """Extract components from structured response.

        Expected format:
        ### SECTION_NAME
        ```lang
        content
        ```
        """
        components: Dict[str, Any] = {}

        # Pattern to match ### SECTION_NAME (optional text) followed by code block
        # Uses .*? to allow optional descriptions like "(optional, if needed)"
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

                # Special handling for tiling_fields (text format)
                if key == "tiling_fields":
                    components[key] = self._parse_tiling_fields(content)
                # Special handling for include lists (JSON or text)
                elif key in ("kernel_includes", "tiling_includes", "tiling_func_includes"):
                    components[key] = self._parse_includes(content)
                else:
                    components[key] = content

        return components

    def _parse_tiling_fields(self, content: str) -> List[Dict[str, Any]]:
        """Parse tiling fields from text format.

        Supported formats:
        - TYPE NAME                           → {"name": NAME, "type": TYPE}
        - TYPE NAME[SIZE]                     → {"name": NAME, "type": TYPE, "size": SIZE}
        - struct TYPE NAME                    → {"name": NAME, "type": TYPE, "is_struct": True}

        Also supports legacy JSON format for backwards compatibility.
        """
        content = content.strip()

        # Try JSON first (backwards compatibility)
        if content.startswith("["):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                fixed = self._fix_json(content)
                if fixed is not None:
                    return fixed

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
        # Pattern for struct without include: struct TYPE NAME
        struct_simple_match = re.match(
            r"struct\s+(\w+)\s+(\w+)$",
            line, re.IGNORECASE
        )
        if struct_simple_match:
            return {
                "name": struct_simple_match.group(2),
                "type": struct_simple_match.group(1),
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
        """Parse include list from JSON or text format.

        Validates that each entry looks like a valid header file path.
        Filters out markdown artifacts like ``` that may leak from LLM output.
        """
        content = content.strip()

        # Filter out markdown code fence artifacts
        if content in ('```', '```cpp', '```c++', '```c', '```h'):
            return []

        # Try JSON first
        if content.startswith("["):
            try:
                parsed = json.loads(content)
                # Validate each entry
                return [h for h in parsed if self._is_valid_header_path(h)]
            except json.JSONDecodeError:
                pass

        # Parse as text (one per line)
        includes = []
        for line in content.split("\n"):
            line = line.strip().strip('"').strip("'")
            # Skip empty lines, comments
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            # Validate header path
            if self._is_valid_header_path(line):
                includes.append(line)
        return includes

    def _is_valid_header_path(self, path: str) -> bool:
        """Check if a string looks like a valid C/C++ header path.

        Valid patterns:
        - Contains .h extension (e.g., "kernel_operator.h", "tiling/platform.h")
        - Contains / path separator (e.g., "lib/matmul_intf.h")

        Invalid patterns:
        - Markdown artifacts (```, ```cpp, etc.)
        - Empty or whitespace only
        - Comment lines
        """
        if not path or not path.strip():
            return False
        # Filter markdown artifacts
        if path.startswith('`') or path.endswith('`'):
            return False
        # Should contain .h or have a path-like structure
        if '.h' in path or '/' in path:
            return True
        # Also accept common header extensions
        if any(path.endswith(ext) for ext in ('.hpp', '.hxx', '.h++')):
            return True
        return False

    def _fix_json(self, content: str) -> Optional[List]:
        """Try to fix common JSON formatting issues."""
        # Remove trailing commas
        content = re.sub(r",\s*([}\]])", r"\1", content)

        # Try parsing again
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def _fallback_parse(self, response_str: str, existing: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback parsing for responses that don't follow the exact format.

        Tries to extract components using alternative patterns.
        """
        result = existing.copy()

        # Try to find code blocks without proper headers
        code_blocks = re.findall(r"```(?:cpp|c\+\+|c)?\s*\n(.*?)\n```", response_str, re.DOTALL)

        # Heuristics to identify component types
        for block in code_blocks:
            block = block.strip()
            if not block:
                continue

            # Skip if looks like JSON
            if block.startswith("[") or block.startswith("{"):
                continue

            # Identify by content patterns
            if "class Kernel" in block and "kernel_impl" not in result:
                result["kernel_impl"] = block
            elif "op.Init" in block and "op.Process" in block and "kernel_entry_body" not in result:
                result["kernel_entry_body"] = block
            elif "TilingData tiling" in block and "tiling_func_body" not in result:
                result["tiling_func_body"] = block
            elif "GetInputShape" in block and "GetOutputShape" in block and "infer_shape_body" not in result:
                result["infer_shape_body"] = block
            elif "at::Tensor result" in block and "output_alloc_code" not in result:
                result["output_alloc_code"] = block

        # Try to find tiling_fields (JSON or text format)
        if "tiling_fields" not in result:
            # Try JSON format first
            json_match = re.search(r"```json\s*\n(\[.*?\])\n```", response_str, re.DOTALL)
            if json_match:
                try:
                    result["tiling_fields"] = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try text format (look for code block with uint32_t/int64_t patterns)
            if "tiling_fields" not in result:
                for block in re.findall(r"```\s*\n(.*?)\n```", response_str, re.DOTALL):
                    block = block.strip()
                    if re.search(r"\b(uint32_t|int64_t|int32_t|float)\s+\w+", block):
                        fields = self._parse_tiling_fields(block)
                        if fields:
                            result["tiling_fields"] = fields
                            break

        return result
