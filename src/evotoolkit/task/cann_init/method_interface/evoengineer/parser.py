# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Response parsing for EvoEngineer CANN Interface"""

import json
import re

from evotoolkit.core import Solution


class ParserMixin:
    """Mixin for response parsing methods"""

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response into Solution with all CANN components"""
        if not response_str or not response_str.strip():
            return Solution("")

        content = response_str.strip()

        # Extract name
        name = self._extract_field(content, r"name:\s*([^\n]+)")

        # Extract thought
        thought = self._extract_field(content, r"thought:\s*([^\n]+)")

        # Extract kernel_src
        kernel_src = self._extract_code_block(content, "kernel_src")

        # Extract tiling_fields (JSON)
        tiling_fields = self._extract_json_block(content, "tiling_fields")

        # Extract tiling_func_body
        tiling_func_body = self._extract_code_block(content, "tiling_func_body")

        # Extract infer_shape_body
        infer_shape_body = self._extract_code_block(content, "infer_shape_body")

        # Fallback: try to extract any cpp code block if kernel_src not found
        if not kernel_src:
            kernel_src = self._extract_any_code_block(content)

        # Assign project path if projects_dir is set
        other_info = {
            "name": name or "generated",
            "thought": thought or "",
            "tiling_fields": tiling_fields or [],
            "tiling_func_body": tiling_func_body or "",
            "infer_shape_body": infer_shape_body or "",
        }

        if self.projects_dir:
            self.solution_counter += 1
            project_path = self.projects_dir / f"solution_{self.solution_counter:04d}"
            other_info["project_path"] = str(project_path)

        return Solution(
            sol_string=kernel_src,
            other_info=other_info,
        )

    def _extract_field(self, content: str, pattern: str) -> str:
        """Extract a simple field value"""
        match = re.search(pattern, content, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_code_block(self, content: str, label: str) -> str:
        """Extract code block after a label"""
        # Pattern: label:\n```cpp\n...\n```
        pattern = rf"{label}:\s*\n*```(?:cpp|c\+\+)?\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_json_block(self, content: str, label: str) -> list:
        """Extract JSON array after a label"""
        pattern = rf"{label}:\s*\n*```(?:json)?\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        return []

    def _extract_any_code_block(self, content: str) -> str:
        """Fallback: extract any code block"""
        pattern = r"```(?:cpp|c\+\+)?\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
