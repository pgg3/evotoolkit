# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Response parsing for EvoEngineer CANN Interface

This module parses LLM responses to extract 6 components for Ascend C operators,
along with name and thought metadata.
"""

import json
import re
from typing import Any, Dict, List, Optional

from evotoolkit.core import Solution


class ParserMixin:
    """Mixin for response parsing methods"""

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response into Solution with all CANN components.

        Extracts:
        - name: Descriptive name
        - thought: Optimization rationale
        - kernel_impl: Kernel class code
        - kernel_entry_body: Entry function body
        - tiling_fields: TilingData fields
        - tiling_func_body: TilingFunc body
        - infer_shape_body: InferShape body
        - output_alloc_code: Output allocation code
        - kernel_includes: Optional kernel headers
        - tiling_includes: Optional tiling headers
        """
        if not response_str or not response_str.strip():
            return Solution("")

        content = response_str.strip()

        # Extract name
        name = self._extract_field(content, r"^name:\s*([^\n]+)")

        # Extract thought
        thought = self._extract_field(content, r"^thought:\s*([^\n]+)")

        # Extract 6 components
        components = self._extract_components(content)

        # Add name and thought
        components["name"] = name or "evoengineer_generated"
        components["thought"] = thought or ""

        # Assign project path if projects_dir is set
        if self.projects_dir:
            self.solution_counter += 1
            project_path = self.projects_dir / f"solution_{self.solution_counter:04d}"
            components["project_path"] = str(project_path)

        return Solution(sol_string="", other_info=components)

    def _extract_field(self, content: str, pattern: str) -> str:
        """Extract a simple field value."""
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_components(self, content: str) -> Dict[str, Any]:
        """Extract 6 components from structured response.

        Expected format:
        ### SECTION_NAME
        ```lang
        content
        ```
        """
        components: Dict[str, Any] = {}

        # Pattern to match ### SECTION_NAME followed by code block
        pattern = r"###\s*(\w+).*?\n```(?:\w+)?\s*\n(.*?)\n```"

        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
            section_name = match.group(1).upper()
            section_content = match.group(2).strip()

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
                    components[key] = self._parse_tiling_fields(section_content)
                # Special handling for include lists
                elif key in ("kernel_includes", "tiling_includes", "tiling_func_includes"):
                    components[key] = self._parse_includes(section_content)
                else:
                    components[key] = section_content

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

    def _fix_json(self, content: str) -> Optional[List]:
        """Try to fix common JSON formatting issues."""
        # Remove trailing commas
        content = re.sub(r",\s*([}\]])", r"\1", content)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
