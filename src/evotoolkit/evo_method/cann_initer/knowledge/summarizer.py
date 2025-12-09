# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Knowledge Summarizer for CANNIniter

Summarizes raw knowledge (API docs, operator examples) into concise context
for Implementation Agents.

Key principles:
- API: Full coverage (correctness depends on it)
- Example: Limited quantity (1-2 most relevant)
"""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .prompts import SUMMARIZER_PROMPT


class KnowledgeSummarizer:
    """Summarize raw knowledge into concise context for Implementation Agent

    Two-stage processing:
    1. API summaries: Extract signatures and descriptions (no LLM needed)
    2. Example summaries: Use LLM to select and extract relevant code
    """

    def __init__(
        self,
        llm_client: Optional[Callable[[str], str]] = None,
        max_examples: int = 2,
        cann_path: Optional[str] = None,
    ):
        """
        Args:
            llm_client: LLM call function, if None only API summaries are generated
            max_examples: Maximum number of examples to include
            cann_path: CANN SDK path for reading header files
        """
        self.llm_client = llm_client
        self.max_examples = max_examples
        self.cann_path = cann_path

    def summarize(
        self,
        task_context: Dict[str, Any],
        raw_knowledge: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Summarize raw knowledge into concise context.

        Args:
            task_context: {
                "operator_description": str,
                "kernel_pseudocode": str,
                "tiling_execution": str,
                "tiling_fields": list,
            }
            raw_knowledge: {
                "apis": {name: search_api() result},
                "examples": {name: search_operator() result},
            }

        Returns:
            {
                "api_summaries": [{"name", "signature", "description"}, ...],
                "example_summaries": [{"name", "purpose", "code_snippet"}, ...],
                "combined_context": str,  # Formatted context for Impl Agent
            }
        """
        # Stage 1: API summaries (no LLM needed)
        api_summaries = self._summarize_apis(raw_knowledge.get("apis", {}))

        # Stage 2: Example summaries (uses LLM if available)
        example_summaries = self._summarize_examples(
            task_context, raw_knowledge.get("examples", {})
        )

        # Combine into final context
        combined_context = self._format_combined_context(
            api_summaries, example_summaries
        )

        return {
            "api_summaries": api_summaries,
            "example_summaries": example_summaries,
            "combined_context": combined_context,
        }

    # =========================================================================
    # API Summarization (Full Coverage)
    # =========================================================================

    def _summarize_apis(self, apis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Summarize all APIs (no LLM needed)"""
        summaries = []
        for api_name, api_result in apis.items():
            if api_result.get("status") != "found":
                continue

            api_info = api_result.get("api_info", {})
            summary = {
                "name": api_name,
                "signature": self._get_api_signature(api_info),
                "description": api_info.get("description", ""),
                "category": api_info.get("category", ""),
                "header": api_info.get("header", ""),
            }
            summaries.append(summary)

        return summaries

    def _get_api_signature(self, api_info: Dict[str, Any]) -> str:
        """Get API signature from header file if available"""
        api_name = api_info.get("name", "")
        header_name = api_info.get("header", "")

        if not header_name or not self.cann_path:
            # Return basic signature
            return f"void {api_name}(LocalTensor<T>& dst, LocalTensor<T>& src, ...)"

        # Try to read from header file
        header_path = self._find_header_path(header_name)
        if not header_path or not header_path.exists():
            return f"void {api_name}(LocalTensor<T>& dst, LocalTensor<T>& src, ...)"

        try:
            content = header_path.read_text(errors="ignore")
            # Find function declaration
            pattern = rf"""
                __aicore__\s+inline\s+          # __aicore__ inline
                (?:__inout_pipe__\([^)]+\)\s+)? # Optional pipe annotation
                (void|[A-Za-z_][A-Za-z0-9_<>]*)\s+  # Return type
                {re.escape(api_name)}\s*        # Function name
                \([^){{]*\)                     # Parameters (non-greedy, stop at ) or {{)
            """
            match = re.search(pattern, content, re.VERBOSE)
            if match:
                sig = match.group(0)
                # Clean up the signature
                sig = re.sub(r"__aicore__\s+inline\s+", "", sig)
                sig = re.sub(r"__inout_pipe__\([^)]+\)\s+", "", sig)
                return sig.strip()
        except Exception:
            pass

        return f"void {api_name}(LocalTensor<T>& dst, LocalTensor<T>& src, ...)"

    def _find_header_path(self, header_name: str) -> Optional[Path]:
        """Find header file path in CANN SDK"""
        if not self.cann_path:
            return None

        cann = Path(self.cann_path)
        candidates = [
            cann / "aarch64-linux" / "ascendc" / "include" / "basic_api" / "interface" / header_name,
            cann / "x86_64-linux" / "ascendc" / "include" / "basic_api" / "interface" / header_name,
            cann / "include" / "ascendc" / "basic_api" / "interface" / header_name,
        ]

        for path in candidates:
            if path.exists():
                return path
        return None

    # =========================================================================
    # Example Summarization (Limited Quantity)
    # =========================================================================

    def _summarize_examples(
        self,
        task_context: Dict[str, Any],
        examples: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Summarize examples (uses LLM if available)"""
        if not examples:
            return []

        # If no LLM, use rule-based extraction
        if not self.llm_client:
            return self._summarize_examples_rule_based(examples)

        # Use LLM to select and summarize
        return self._summarize_examples_with_llm(task_context, examples)

    def _summarize_examples_rule_based(
        self, examples: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rule-based example summarization (no LLM)"""
        summaries = []
        count = 0

        for name, result in examples.items():
            if count >= self.max_examples:
                break

            primary = result.get("primary")
            if not primary:
                continue

            # Extract core kernel logic
            kernel_core = self._extract_kernel_core(
                primary.get("kernel_code") or "", max_lines=60
            )

            # Extract core tiling logic
            tiling_core = self._extract_tiling_core(
                primary.get("host_code") or "", max_lines=40
            )

            summary = {
                "name": name,
                "purpose": self._extract_purpose_from_readme(primary.get("readme") or ""),
                "key_techniques": [],
                "kernel_snippet": kernel_core,
                "tiling_snippet": tiling_core,
            }
            summaries.append(summary)
            count += 1

        return summaries

    def _summarize_examples_with_llm(
        self,
        task_context: Dict[str, Any],
        examples: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Use LLM to select and summarize examples"""
        # Prepare examples content
        examples_content = self._format_examples_for_llm(examples)

        # Build prompt
        prompt = SUMMARIZER_PROMPT.format(
            operator_description=task_context.get("operator_description", ""),
            kernel_pseudocode=task_context.get("kernel_pseudocode", ""),
            examples_content=examples_content,
            max_examples=self.max_examples,
        )

        # Call LLM
        try:
            response = self.llm_client(prompt)
            return self._parse_example_summaries(response)
        except Exception as e:
            print(f"[KnowledgeSummarizer] LLM call failed: {e}")
            return self._summarize_examples_rule_based(examples)

    def _format_examples_for_llm(self, examples: Dict[str, Any]) -> str:
        """Format examples for LLM prompt

        包含精简后的 kernel 和 tiling 代码
        """
        parts = []
        for name, result in examples.items():
            primary = result.get("primary")
            if not primary:
                continue

            readme = (primary.get("readme") or "")[:300]  # Truncate README

            # Extract core logic only
            kernel_core = self._extract_kernel_core(
                primary.get("kernel_code") or "", max_lines=60
            )
            tiling_core = self._extract_tiling_core(
                primary.get("host_code") or "", max_lines=40
            )

            parts.append(f"""
### {name}
**README**: {readme}

**Kernel Code** (core logic):
```cpp
{kernel_core}
```

**Tiling Code** (core logic):
```cpp
{tiling_core}
```
""")
        return "\n".join(parts)

    def _parse_example_summaries(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for example summaries"""
        summaries = []

        # Extract content between <example_summaries> tags
        match = re.search(
            r"<example_summaries>(.*?)</example_summaries>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            # Try to parse without tags
            content = response
        else:
            content = match.group(1)

        # Parse each example section
        example_pattern = r"###\s+(\S+)\s*\n(.*?)(?=###|\Z)"
        for match in re.finditer(example_pattern, content, re.DOTALL):
            name = match.group(1)
            section = match.group(2)

            summary = {
                "name": name,
                "purpose": self._extract_field(section, "理由|相关度|目的"),
                "key_techniques": self._extract_list_field(section, "关键技术"),
                "code_snippet": self._extract_code_block(section),
            }
            summaries.append(summary)

            if len(summaries) >= self.max_examples:
                break

        return summaries

    def _extract_field(self, text: str, field_pattern: str) -> str:
        """Extract a field value from text"""
        pattern = rf"\*\*({field_pattern})\*\*:\s*(.+?)(?=\n\*\*|\n###|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(2).strip()
        return ""

    def _extract_list_field(self, text: str, field_name: str) -> List[str]:
        """Extract a list field from text"""
        pattern = rf"\*\*{field_name}\*\*:\s*\n((?:[-*]\s+.+\n?)+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            items = re.findall(r"[-*]\s+(.+)", match.group(1))
            return items
        return []

    def _extract_code_block(self, text: str) -> str:
        """Extract code block from text"""
        match = re.search(r"```(?:cpp)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_purpose_from_readme(self, readme: str) -> str:
        """Extract purpose from README"""
        if not readme:
            return ""

        # Try to find first meaningful paragraph
        lines = readme.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 20:
                return line[:200]
        return ""

    def _extract_code_snippet(self, kernel_code: str, max_lines: int = 80) -> str:
        """Extract most relevant code snippet (deprecated, use _extract_kernel_core)"""
        return self._extract_kernel_core(kernel_code, max_lines)

    # =========================================================================
    # Code Extraction (Core Logic Only)
    # =========================================================================

    def _extract_kernel_core(self, kernel_code: str, max_lines: int = 60) -> str:
        """Extract core kernel logic, removing boilerplate

        精简规则:
        1. 删除 #include, using namespace 等样板代码
        2. 删除类成员变量声明
        3. 只保留核心函数: Init(), Process(), Compute(), CopyIn(), CopyOut()
        4. 限制总行数
        """
        if not kernel_code:
            return ""

        lines = kernel_code.split("\n")
        result_lines = []
        in_target_func = False
        brace_depth = 0
        target_funcs = ["Init", "Process", "Compute", "CopyIn", "CopyOut", "Calculate"]

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip boilerplate
            if self._is_boilerplate(stripped):
                i += 1
                continue

            # Check if entering a target function
            if not in_target_func:
                for func in target_funcs:
                    # Match function definition: __aicore__ inline void Process(...) or void Process(...)
                    if re.search(rf'\b{func}\s*\(', line) and '{' in line:
                        in_target_func = True
                        brace_depth = line.count('{') - line.count('}')
                        result_lines.append(f"// === {func} ===")
                        result_lines.append(line)
                        break
                    elif re.search(rf'\b{func}\s*\(', line):
                        # Function signature on this line, { on next line
                        in_target_func = True
                        brace_depth = 0
                        result_lines.append(f"// === {func} ===")
                        result_lines.append(line)
                        break
            else:
                # Inside target function
                brace_depth += line.count('{') - line.count('}')
                result_lines.append(line)

                if brace_depth <= 0:
                    in_target_func = False
                    result_lines.append("")  # Add blank line between functions

            i += 1

            # Limit total lines
            if len(result_lines) >= max_lines:
                result_lines.append("// ... (truncated)")
                break

        return "\n".join(result_lines)

    def _extract_tiling_core(self, host_code: str, max_lines: int = 40) -> str:
        """Extract core tiling logic from host code

        精简规则:
        1. 删除 #include, namespace, 注册宏等样板
        2. 只保留 TilingFunc 或 tiling 计算函数
        3. 限制总行数
        """
        if not host_code:
            return ""

        lines = host_code.split("\n")
        result_lines = []
        in_tiling_func = False
        brace_depth = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip boilerplate
            if self._is_boilerplate(stripped):
                i += 1
                continue

            # Skip registration macros
            if stripped.startswith("REGISTER_") or stripped.startswith("REG_OP"):
                i += 1
                continue

            # Look for tiling function
            if not in_tiling_func:
                # Match: TilingFunc, xxxTilingFunc, or functions containing "Tiling"
                if re.search(r'\bTilingFunc\b|\bTiling\w*\s*\(|tiling\w*\s*\(', line, re.IGNORECASE):
                    if '{' in line or (i + 1 < len(lines) and '{' in lines[i + 1]):
                        in_tiling_func = True
                        brace_depth = line.count('{') - line.count('}')
                        result_lines.append("// === Tiling ===")
                        result_lines.append(line)
            else:
                brace_depth += line.count('{') - line.count('}')
                result_lines.append(line)

                if brace_depth <= 0:
                    in_tiling_func = False
                    break  # Usually only need one tiling function

            i += 1

            if len(result_lines) >= max_lines:
                result_lines.append("// ... (truncated)")
                break

        return "\n".join(result_lines)

    def _is_boilerplate(self, line: str) -> bool:
        """Check if line is boilerplate code"""
        boilerplate_patterns = [
            r'^#include',
            r'^#pragma',
            r'^#ifndef',
            r'^#define',
            r'^#endif',
            r'^using\s+namespace',
            r'^namespace\s+\w+',
            r'^\s*$',  # Empty lines at start
            r'^//',    # Comment-only lines (at file level)
            r'^\s*\*', # Multi-line comment continuation
            r'^/\*',   # Multi-line comment start
            r'^\*/',   # Multi-line comment end
            r'^extern\s+"C"',
            r'^class\s+\w+\s*;',  # Forward declaration
        ]
        for pattern in boilerplate_patterns:
            if re.match(pattern, line):
                return True
        return False

    # =========================================================================
    # Combined Context Formatting
    # =========================================================================

    def _format_combined_context(
        self,
        api_summaries: List[Dict[str, Any]],
        example_summaries: List[Dict[str, Any]],
    ) -> str:
        """Format combined context for Implementation Agent"""
        parts = []

        # API Reference section
        if api_summaries:
            parts.append("## API Reference\n")
            for api in api_summaries:
                parts.append(f"### {api['name']}")
                parts.append(f"- **签名**: `{api['signature']}`")
                if api.get("description"):
                    parts.append(f"- **描述**: {api['description']}")
                parts.append("")

        # Example Reference section
        if example_summaries:
            parts.append("## Example Reference\n")
            for ex in example_summaries:
                parts.append(f"### {ex['name']}")
                if ex.get("purpose"):
                    parts.append(f"**目的**: {ex['purpose']}\n")

                if ex.get("key_techniques"):
                    parts.append("**关键技术**:")
                    for tech in ex["key_techniques"]:
                        parts.append(f"- {tech}")
                    parts.append("")

                # Kernel snippet (精简后的核心逻辑)
                kernel_snippet = ex.get("kernel_snippet") or ex.get("code_snippet")
                if kernel_snippet:
                    parts.append("**Kernel 参考代码** (核心逻辑):")
                    parts.append("```cpp")
                    parts.append(kernel_snippet)
                    parts.append("```")

                # Tiling snippet (精简后的 tiling 逻辑)
                tiling_snippet = ex.get("tiling_snippet")
                if tiling_snippet:
                    parts.append("\n**Tiling 参考代码** (核心逻辑):")
                    parts.append("```cpp")
                    parts.append(tiling_snippet)
                    parts.append("```")

                parts.append("")

        return "\n".join(parts)
