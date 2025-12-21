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

from evotoolkit.task.cann_init.method_interface.prompts.phase0 import (
    _format_signature_for_kernel,
)
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
                "operator_signature": dict or Signature object,
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
        """Get API signatures from header file - returns top 5 simplest overloads

        Returns up to 5 overloaded versions, sorted by parameter count (simplest first).
        Format: multi-line with numbered signatures.
        """
        MAX_SIGNATURES = 5  # 显示前 5 个最简单的重载

        api_name = api_info.get("name", "")
        header_name = api_info.get("header", "")

        if not header_name or not self.cann_path:
            return f"void {api_name}(LocalTensor<T>& dst, LocalTensor<T>& src, ...)"

        header_path = self._find_header_path(header_name)
        if not header_path or not header_path.exists():
            return f"void {api_name}(LocalTensor<T>& dst, LocalTensor<T>& src, ...)"

        try:
            content = header_path.read_text(errors="ignore")

            # Find ALL function declarations for this API
            pattern = rf"""
                __aicore__\s+inline\s+          # __aicore__ inline
                (?:__inout_pipe__\([^)]+\)\s+)? # Optional pipe annotation
                (void|[A-Za-z_][A-Za-z0-9_<>]*)\s+  # Return type
                {re.escape(api_name)}\s*        # Function name
                \([^){{]*\)                     # Parameters
            """
            matches = re.findall(pattern, content, re.VERBOSE)

            if not matches:
                return f"void {api_name}(LocalTensor<T>& dst, LocalTensor<T>& src, ...)"

            # Extract full signatures and clean them up
            signatures = []
            for match in re.finditer(pattern, content, re.VERBOSE):
                sig = match.group(0)
                # Clean up
                sig = re.sub(r"__aicore__\s+inline\s+", "", sig)
                sig = re.sub(r"__inout_pipe__\([^)]+\)\s+", "", sig)
                # Normalize whitespace (remove excessive spaces/newlines)
                sig = re.sub(r'\s+', ' ', sig).strip()
                # Count parameters (by counting commas + 1, roughly)
                param_count = sig.count(',') + 1
                signatures.append((param_count, sig))

            # Remove duplicates and sort by parameter count (simplest first)
            seen = set()
            unique_sigs = []
            for _, sig in sorted(signatures, key=lambda x: x[0]):
                if sig not in seen:
                    seen.add(sig)
                    unique_sigs.append(sig)

            total = len(unique_sigs)
            if total == 1:
                return unique_sigs[0]
            elif total == 2:
                # 2 个版本：简化 vs 完整
                return f"(1) {unique_sigs[0]}\n(2) {unique_sigs[1]}"
            else:
                # 多个版本：显示前 MAX_SIGNATURES 个
                result_lines = []
                for i, sig in enumerate(unique_sigs[:MAX_SIGNATURES], 1):
                    result_lines.append(f"({i}) {sig}")
                if total > MAX_SIGNATURES:
                    result_lines.append(f"... (共 {total} 个重载)")
                return "\n".join(result_lines)

        except Exception:
            pass

        return f"void {api_name}(LocalTensor<T>& dst, LocalTensor<T>& src, ...)"

    def _find_header_path(self, header_name: str) -> Optional[Path]:
        """Find header file path in CANN SDK"""
        if not self.cann_path:
            return None

        cann = Path(self.cann_path)
        candidates = [
            # New structure (CANN 8.x+): compiler/ascendc/...
            cann / "compiler" / "ascendc" / "include" / "basic_api" / "interface" / header_name,
            # Legacy structure: arch-specific
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

        # Format operator signature (与 kernel_prompts.py 一致)
        operator_sig = task_context.get("operator_signature")
        formatted_sig = _format_signature_for_kernel(operator_sig) \
            if operator_sig else "Not provided"

        # Format tiling fields for display
        tiling_fields = task_context.get("tiling_fields", [])
        if isinstance(tiling_fields, list) and tiling_fields:
            try:
                tiling_fields_str = "\n".join(
                    f"- {f.get('name', 'unknown')}: {f.get('type', 'unknown')} - {f.get('purpose', '')}"
                    if isinstance(f, dict) else str(f)
                    for f in tiling_fields
                )
            except Exception:
                tiling_fields_str = str(tiling_fields)
        elif tiling_fields:
            tiling_fields_str = str(tiling_fields)
        else:
            tiling_fields_str = "None"

        # Build prompt
        prompt = SUMMARIZER_PROMPT.format(
            operator_signature=formatted_sig,
            kernel_pseudocode=task_context.get("kernel_pseudocode", ""),
            tiling_execution=task_context.get("tiling_execution", ""),
            tiling_fields=tiling_fields_str,
            examples_content=examples_content,
            max_examples=self.max_examples,
        )

        # Call LLM
        try:
            response = self.llm_client(prompt)
            if not response or not isinstance(response, str):
                print(f"[KnowledgeSummarizer] LLM returned invalid response: {type(response)}")
                return self._summarize_examples_rule_based(examples)
            return self._parse_example_summaries(response)
        except Exception as e:
            import traceback
            print(f"[KnowledgeSummarizer] LLM call failed: {e}")
            print(f"[KnowledgeSummarizer] Traceback: {traceback.format_exc()}")
            return self._summarize_examples_rule_based(examples)

    def _format_examples_for_llm(self, examples: Dict[str, Any]) -> str:
        """Format examples for LLM prompt

        只包含有实际代码的 example，不包含 README（对实现没帮助）
        限制最多 3 个高质量 example 以控制 token 数量
        """
        MAX_EXAMPLES = 3
        MIN_KERNEL_LINES = 5  # 至少 5 行才算有效代码（放宽限制）

        parts = []
        example_count = 0

        for name, result in examples.items():
            if example_count >= MAX_EXAMPLES:
                break

            primary = result.get("primary")
            if not primary:
                continue

            # Get raw kernel code first
            raw_kernel = primary.get("kernel_code") or ""

            # Extract core logic only
            kernel_core = self._extract_kernel_core(raw_kernel, max_lines=100)
            tiling_core = self._extract_tiling_core(
                primary.get("host_code") or "", max_lines=60
            )

            # Skip examples with insufficient kernel code
            kernel_lines = len([l for l in kernel_core.split('\n') if l.strip()])
            raw_lines = len([l for l in raw_kernel.split('\n') if l.strip()]) if raw_kernel else 0

            if kernel_lines < MIN_KERNEL_LINES:
                # Fallback: use raw kernel code if extraction failed
                if raw_lines >= MIN_KERNEL_LINES:
                    # Take first portion of raw kernel (skip includes at top)
                    raw_lines_list = raw_kernel.split('\n')
                    # Find first non-boilerplate line
                    start_idx = 0
                    for idx, line in enumerate(raw_lines_list):
                        stripped = line.strip()
                        if not self._is_boilerplate(stripped) and stripped:
                            start_idx = max(0, idx - 2)  # Include 2 lines before
                            break
                    kernel_core = '\n'.join(raw_lines_list[start_idx:start_idx + 100])
                else:
                    print(f"[KnowledgeSummarizer] Skipping example '{name}': kernel_lines={kernel_lines}, raw_lines={raw_lines}")
                    continue

            parts.append(f"""
### {name}

**Kernel Code**:
```cpp
{kernel_core}
```

**Tiling Code**:
```cpp
{tiling_core}
```
""")
            example_count += 1

        return "\n".join(parts)

    def _parse_example_summaries(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for example summaries"""
        summaries = []

        # Handle None or non-string responses
        if not response or not isinstance(response, str):
            return summaries

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

            # New format: extract complete code patterns
            summary = {
                "name": name,
                "purpose": self._extract_field(section, "Why Selected"),
                "adaptation_notes": self._extract_field(section, "Adaptation Notes"),
                # Code patterns - extract complete code blocks
                "init_pattern": self._extract_labeled_code_block(section, "Init Pattern"),
                "process_pattern": self._extract_labeled_code_block(section, "Process Pattern"),
                "compute_pattern": self._extract_labeled_code_block(section, "Compute Pattern"),
                "copyinout_pattern": self._extract_labeled_code_block(section, "CopyIn/CopyOut Pattern"),
                # Key API calls - critical for correct implementation
                "key_api_calls": self._extract_labeled_code_block(section, "Key API Calls"),
                # Legacy fields for backward compatibility
                "kernel_snippet": self._extract_labeled_code_block(section, "Kernel"),
                "tiling_snippet": self._extract_labeled_code_block(section, "Tiling"),
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
            captured = match.group(2)
            if captured and isinstance(captured, str):
                return captured.strip()
        return ""

    def _extract_list_field(self, text: str, field_name: str) -> List[str]:
        """Extract a list field from text"""
        pattern = rf"\*\*{field_name}\*\*:\s*\n((?:[-*]\s+.+\n?)+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            captured = match.group(1)
            if captured and isinstance(captured, str):
                items = re.findall(r"[-*]\s+(.+)", captured)
                return items
        return []

    def _extract_code_block(self, text: str) -> str:
        """Extract code block from text"""
        match = re.search(r"```(?:cpp)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            captured = match.group(1)
            if captured and isinstance(captured, str):
                return captured.strip()
        return ""

    def _extract_labeled_code_block(self, text: str, label: str) -> str:
        """Extract code block with a specific label (e.g., Kernel, Init Pattern)

        Looks for patterns like:
        **Init Pattern** (buffer setup):
        ```cpp
        ...
        ```

        Or legacy format:
        **Kernel Reference Code**:
        ```cpp
        ...
        ```
        """
        # Pattern 1: **Label** (description): - new format
        # e.g., **Init Pattern** (buffer setup):
        pattern1 = rf"\*\*{label}\*\*[^:]*:\s*\n```(?:cpp)?\s*\n(.*?)```"
        match = re.search(pattern1, text, re.DOTALL | re.IGNORECASE)
        if match:
            captured = match.group(1)
            if captured and isinstance(captured, str):
                return captured.strip()

        # Pattern 2: **Label ... Code**: - legacy format
        # e.g., **Kernel Reference Code**:
        pattern2 = rf"\*\*{label}[^*]*\*\*:\s*\n```(?:cpp)?\s*\n(.*?)```"
        match = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)
        if match:
            captured = match.group(1)
            if captured and isinstance(captured, str):
                return captured.strip()

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

    def _extract_kernel_core(self, kernel_code: str, max_lines: int = 100) -> str:
        """Extract core kernel logic, removing boilerplate

        精简规则:
        1. 删除 #include, using namespace 等样板代码
        2. 只提取函数定义（有实现的），跳过声明和调用
        3. 只保留核心函数: Init(), Process(), Compute(), CopyIn(), CopyOut()
        4. 每个函数只提取一次（优先类外定义 ClassName::Func）
        5. 限制总行数
        """
        if not kernel_code:
            return ""

        lines = kernel_code.split("\n")
        result_lines = []
        in_target_func = False
        brace_depth = 0
        extracted_funcs = set()  # Track which functions we've already extracted
        # Include common variants: CopyInAndCast, CastAndCopyOut, etc.
        target_funcs = [
            "Init", "Process", "Compute", "Calculate",
            "CopyIn", "CopyOut",
            "CopyInAndCast", "CastAndCopyOut",  # Common variants
            "DataCopyIn", "DataCopyOut",
        ]

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip boilerplate
            if self._is_boilerplate(stripped):
                i += 1
                continue

            # Check if entering a target function DEFINITION (not call or declaration)
            if not in_target_func:
                for func in target_funcs:
                    # Skip if already extracted this function
                    if func in extracted_funcs:
                        continue

                    # Pattern for function DEFINITION (not call):
                    # - Must NOT start with obj.Func or obj->Func (that's a call)
                    # - Must have return type or __aicore__ before function name
                    # - Must have { on this line or next line

                    # Skip function calls: xxx.Init( or xxx->Init(
                    if re.search(rf'\.\s*{func}\s*\(', line) or re.search(rf'->\s*{func}\s*\(', line):
                        continue

                    # Skip single-line calls like: op.Init(...); op.Process();
                    if re.search(rf'\b\w+\.\s*{func}\s*\([^)]*\)\s*;', line):
                        continue

                    # Match function definition patterns:
                    # 1. ClassName::Func(...) {   (class-external definition)
                    # 2. __aicore__ inline void Func(...) {   (inline definition)
                    # 3. void Func(...) {   (simple definition)
                    is_definition = False

                    # Pattern 1: ClassName::Func(
                    if re.search(rf'\w+::{func}\s*\(', line):
                        is_definition = True
                    # Pattern 2: __aicore__ ... Func(
                    elif re.search(rf'__aicore__.*\b{func}\s*\(', line):
                        is_definition = True
                    # Pattern 3: return_type Func( - but NOT a call
                    elif re.search(rf'\b(void|bool|int|auto)\s+{func}\s*\(', line):
                        is_definition = True

                    if is_definition:
                        # Check for { on this line or next
                        has_brace = '{' in line
                        if not has_brace and i + 1 < len(lines) and '{' in lines[i + 1]:
                            has_brace = True

                        # Skip declarations (end with ; without {)
                        if not has_brace and ';' in line:
                            continue

                        if has_brace:
                            in_target_func = True
                            brace_depth = line.count('{') - line.count('}')
                            result_lines.append(f"// === {func} ===")
                            result_lines.append(line)
                            extracted_funcs.add(func)
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

    def _extract_tiling_core(self, host_code: str, max_lines: int = 60) -> str:
        """Extract core tiling logic from host code

        精简规则:
        1. 删除 #include, namespace, 注册宏等样板
        2. 提取 tiling 函数实现（RunTiling, TilingFunc, TilingPrepare 等）
        3. 限制总行数
        """
        if not host_code:
            return ""

        lines = host_code.split("\n")
        result_lines = []
        in_tiling_func = False
        brace_depth = 0
        func_count = 0

        # Target function patterns (order matters - more specific first)
        # These are actual tiling computation functions
        tiling_func_patterns = [
            r'\bRunBigKernelTiling\s*\(',
            r'\bRunTiling\s*\(',
            r'\bTilingPrepare\s*\(',
            r'\bTilingFunc\s*\(',
            r'\bGetTiling\s*\(',
            # Generic pattern for functions with "tiling" in name followed by implementation
            r'\b\w*[Tt]iling\w*\s*\([^;{]*\)\s*\{',
        ]

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

            # Skip constructor declarations (single line with ; at end)
            if 'explicit' in line and ';' in line:
                i += 1
                continue

            # Look for tiling function implementation
            if not in_tiling_func:
                for pattern in tiling_func_patterns:
                    if re.search(pattern, line):
                        # Check if this is a function definition (has { nearby)
                        has_brace = '{' in line
                        if not has_brace and i + 1 < len(lines):
                            has_brace = '{' in lines[i + 1]

                        if has_brace:
                            in_tiling_func = True
                            brace_depth = line.count('{') - line.count('}')
                            result_lines.append(f"// === Tiling Function {func_count + 1} ===")
                            result_lines.append(line)
                            break
            else:
                brace_depth += line.count('{') - line.count('}')
                result_lines.append(line)

                if brace_depth <= 0:
                    in_tiling_func = False
                    func_count += 1
                    result_lines.append("")
                    # Continue to find more tiling functions (up to 2)
                    if func_count >= 2:
                        break

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
        """Format combined context for Implementation Agent

        Outputs complete, reusable code patterns from examples.
        """
        parts = []

        # API Reference section
        if api_summaries:
            parts.append("## API Reference\n")
            for api in api_summaries:
                parts.append(f"### {api['name']}")
                parts.append(f"- **Signature**: `{api['signature']}`")
                if api.get("description"):
                    parts.append(f"- **Description**: {api['description']}")
                parts.append("")

        # Example Reference section - now outputs complete code patterns
        if example_summaries:
            parts.append("## Example Reference\n")
            for ex in example_summaries:
                parts.append(f"### {ex['name']}")

                # Why this example was selected
                if ex.get("purpose"):
                    parts.append(f"**Why Selected**: {ex['purpose']}\n")

                # Adaptation notes for current task
                if ex.get("adaptation_notes"):
                    parts.append(f"**Adaptation Notes**: {ex['adaptation_notes']}\n")

                # Init Pattern (buffer setup)
                if ex.get("init_pattern"):
                    parts.append("**Init Pattern** (buffer setup):")
                    parts.append("```cpp")
                    parts.append(ex["init_pattern"])
                    parts.append("```\n")

                # Process Pattern (main loop)
                if ex.get("process_pattern"):
                    parts.append("**Process Pattern** (main loop):")
                    parts.append("```cpp")
                    parts.append(ex["process_pattern"])
                    parts.append("```\n")

                # Compute Pattern (core computation)
                if ex.get("compute_pattern"):
                    parts.append("**Compute Pattern** (core computation):")
                    parts.append("```cpp")
                    parts.append(ex["compute_pattern"])
                    parts.append("```\n")

                # CopyIn/CopyOut Pattern (data movement)
                if ex.get("copyinout_pattern"):
                    parts.append("**CopyIn/CopyOut Pattern** (data movement):")
                    parts.append("```cpp")
                    parts.append(ex["copyinout_pattern"])
                    parts.append("```\n")

                # Key API Calls - critical for correct implementation
                if ex.get("key_api_calls"):
                    parts.append("**Key API Calls** (use these exact patterns):")
                    parts.append("```cpp")
                    parts.append(ex["key_api_calls"])
                    parts.append("```\n")

                # Legacy fallback: kernel_snippet and tiling_snippet
                # (for backward compatibility with rule-based extraction)
                if not any([ex.get("init_pattern"), ex.get("process_pattern"),
                           ex.get("compute_pattern"), ex.get("copyinout_pattern")]):
                    kernel_snippet = ex.get("kernel_snippet") or ex.get("code_snippet")
                    if kernel_snippet:
                        parts.append("**Kernel Reference Code**:")
                        parts.append("```cpp")
                        parts.append(kernel_snippet)
                        parts.append("```\n")

                    tiling_snippet = ex.get("tiling_snippet")
                    if tiling_snippet:
                        parts.append("**Tiling Reference Code**:")
                        parts.append("```cpp")
                        parts.append(tiling_snippet)
                        parts.append("```\n")

                parts.append("")

        return "\n".join(parts)
