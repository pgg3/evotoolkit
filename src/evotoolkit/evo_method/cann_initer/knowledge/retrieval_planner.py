# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Retrieval Planner

将 Phase 1 的概念性检索请求转换为精确检索请求。

两种模式：
- LLM 模式：使用 prompt 让 LLM 智能规划
- 规则模式：无 LLM 时的 fallback，使用知识库精确/模糊匹配
"""

import re
from typing import TYPE_CHECKING, Any, Callable, List

from evotoolkit.task.cann_init.method_interface.prompts.phase0 import (
    _format_signature_for_kernel,
)
from .prompts import RETRIEVAL_PLANNER_PROMPT

if TYPE_CHECKING:
    from .knowledge_base import RealKnowledgeBase


class RetrievalPlanner:
    """知识检索规划 Agent

    接管 Phase 1 的概念性检索请求，结合设计信息和知识库摘要，
    输出精确的检索请求列表。

    输入（轻量化）：
    - operator_signature: 算子签名 (dict 或 Signature 对象)
    - kernel_pseudocode: kernel 伪代码
    - tiling_execution: tiling 执行伪代码
    - tiling_fields: tiling 字段列表
    - raw_requests: Phase 1 的概念性请求

    输出：
    - api_requests: 精确的 API 请求
    - example_requests: 精确的 Example 请求
    - skipped: 跳过的请求（带原因）
    - analysis: 决策说明
    """

    def __init__(
        self,
        knowledge_base: "RealKnowledgeBase",
        llm_client: Callable[[str], str] = None,
    ):
        """
        Args:
            knowledge_base: 知识库实例
            llm_client: LLM 调用函数，签名为 (prompt: str) -> str
                        如果为 None，则使用简单的规则匹配
        """
        self.kb = knowledge_base
        self.llm_client = llm_client

    def plan(
        self,
        operator_signature: Any,
        kernel_pseudocode: str,
        tiling_execution: str,
        tiling_fields: List[dict],
        raw_requests: List[dict],
    ) -> dict:
        """规划精确的检索请求

        Args:
            operator_signature: 算子签名 (dict 或 Signature 对象)
            kernel_pseudocode: kernel 伪代码
            tiling_execution: tiling 执行伪代码
            tiling_fields: tiling 字段列表 [{"name", "type", "purpose"}, ...]
            raw_requests: Phase 1 的概念性请求 [{"type": "api"|"example", "name": str}, ...]

        Returns:
            {
                "api_requests": [{"name", "reason", "priority"}, ...],
                "example_requests": [{"name", "reason", "priority"}, ...],
                "skipped": [{"original", "type", "reason"}, ...],
                "analysis": str
            }
        """
        if self.llm_client:
            return self._plan_with_llm(
                operator_signature, kernel_pseudocode, tiling_execution,
                tiling_fields, raw_requests
            )
        else:
            return self._plan_with_rules(raw_requests)

    def _plan_with_llm(
        self,
        operator_signature: Any,
        kernel_pseudocode: str,
        tiling_execution: str,
        tiling_fields: List[dict],
        raw_requests: List[dict],
    ) -> dict:
        """使用 LLM 进行智能规划"""
        # 格式化 operator signature (与 kernel_prompts.py 一致)
        formatted_sig = _format_signature_for_kernel(operator_signature) \
            if operator_signature else "Not provided"

        # 格式化 tiling fields
        fields_str = "\n".join([
            f"- {f['name']}: {f.get('type', 'unknown')} // {f.get('purpose', '')}"
            for f in tiling_fields
        ]) if tiling_fields else "None"

        # 格式化 raw requests
        requests_str = "\n".join([
            f"- [{r['type']}] {r['name']}"
            for r in raw_requests
        ]) if raw_requests else "None"

        # 构建 prompt
        prompt = RETRIEVAL_PLANNER_PROMPT.format(
            operator_signature=formatted_sig,
            kernel_pseudocode=kernel_pseudocode or "Not provided",
            tiling_execution=tiling_execution or "Not provided",
            tiling_fields=fields_str,
            available_knowledge=self.kb.get_available_knowledge_summary(),
            raw_requests=requests_str,
        )

        # 调用 LLM
        response = self.llm_client(prompt)

        # 解析输出
        return self._parse_llm_response(response)

    def _parse_llm_response(self, response: str) -> dict:
        """解析 LLM 输出（Markdown 格式）

        格式：
        <retrieval_plan>
        ## API Requests
        - NAME [PRIORITY]: REASON

        ## Example Requests
        - NAME [PRIORITY]: REASON

        ## Skipped
        - [TYPE] ORIGINAL: REASON

        ## Analysis
        ...
        </retrieval_plan>
        """
        # 提取 <retrieval_plan> 块
        start = response.find('<retrieval_plan>')
        end = response.find('</retrieval_plan>')

        if start == -1 or end == -1:
            return self._empty_result("Failed to find <retrieval_plan> block")

        content = response[start + len('<retrieval_plan>'):end].strip()

        # 解析各个部分
        api_requests = self._parse_request_section(content, "API Requests")
        example_requests = self._parse_request_section(content, "Example Requests")
        skipped = self._parse_skipped_section(content)
        analysis = self._parse_analysis_section(content)

        return {
            "api_requests": api_requests,
            "example_requests": example_requests,
            "skipped": skipped,
            "analysis": analysis,
        }

    def _parse_request_section(self, content: str, section_name: str) -> List[dict]:
        """解析 API Requests 或 Example Requests 部分

        格式: - NAME [PRIORITY]: REASON
        """
        results = []

        # 找到该部分
        pattern = rf"## {section_name}\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return results

        section_content = match.group(1).strip()
        if section_content.lower() == "none":
            return results

        # 解析每一行: - NAME [PRIORITY]: REASON
        line_pattern = r"^- (\S+)\s*\[(\w+)\]:\s*(.+)$"
        for line in section_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            line_match = re.match(line_pattern, line)
            if line_match:
                results.append({
                    "name": line_match.group(1),
                    "priority": line_match.group(2).lower(),
                    "reason": line_match.group(3).strip(),
                })

        return results

    def _parse_skipped_section(self, content: str) -> List[dict]:
        """解析 Skipped 部分

        格式: - [TYPE] ORIGINAL: REASON
        """
        results = []

        pattern = r"## Skipped\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return results

        section_content = match.group(1).strip()
        if section_content.lower() == "none":
            return results

        # 解析每一行: - [TYPE] ORIGINAL: REASON
        line_pattern = r"^- \[(\w+)\]\s*(\S+):\s*(.+)$"
        for line in section_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            line_match = re.match(line_pattern, line)
            if line_match:
                results.append({
                    "type": line_match.group(1).lower(),
                    "original": line_match.group(2),
                    "reason": line_match.group(3).strip(),
                })

        return results

    def _parse_analysis_section(self, content: str) -> str:
        """解析 Analysis 部分"""
        pattern = r"## Analysis\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    # Kernel class method names - NOT system APIs
    KERNEL_METHOD_BLACKLIST = {
        # Lifecycle methods
        "Init", "Process", "Compute", "Calculate",
        # Data movement methods
        "CopyIn", "CopyOut", "CopyInAndCast", "CastAndCopyOut",
        "DataCopyIn", "DataCopyOut",
        # Other common method names
        "Run", "Execute", "Setup", "Cleanup",
    }

    def _plan_with_rules(self, raw_requests: List[dict]) -> dict:
        """使用规则进行简单匹配（无 LLM 时的 fallback）"""
        api_requests = []
        example_requests = []
        skipped = []

        seen_apis = set()
        seen_examples = set()

        for req in raw_requests:
            req_type = req.get("type")
            name = req.get("name", "")

            if req_type == "api":
                # Skip kernel class method names (not system APIs)
                if name in self.KERNEL_METHOD_BLACKLIST:
                    skipped.append({
                        "original": name,
                        "type": "api",
                        "reason": "Kernel class method, not a system API"
                    })
                    continue

                # 检查 API 是否存在
                result = self.kb.search_api(name)
                if result["status"] == "found":
                    if name not in seen_apis:
                        api_requests.append({
                            "name": name,
                            "reason": "Exact match",
                            "priority": "high"
                        })
                        seen_apis.add(name)
                elif result["status"] == "ambiguous" and result["candidates"]:
                    # 使用第一个候选
                    candidate = result["candidates"][0]
                    if candidate not in seen_apis:
                        api_requests.append({
                            "name": candidate,
                            "reason": f"Closest match for '{name}'",
                            "priority": "medium"
                        })
                        seen_apis.add(candidate)
                else:
                    skipped.append({
                        "original": name,
                        "type": "api",
                        "reason": "Not found in available APIs"
                    })

            elif req_type == "example":
                # 检查 Example 是否存在
                result = self.kb.search_operator(name)
                if result["primary"]:
                    op_name = result["primary"]["name"]
                    if op_name not in seen_examples:
                        example_requests.append({
                            "name": op_name,
                            "reason": f"Match for '{name}' (confidence: {result['confidence']})",
                            "priority": "high" if result["confidence"] == "high" else "medium"
                        })
                        seen_examples.add(op_name)
                else:
                    skipped.append({
                        "original": name,
                        "type": "example",
                        "reason": "Not found in available operators"
                    })

        return {
            "api_requests": api_requests,
            "example_requests": example_requests,
            "skipped": skipped,
            "analysis": "Rule-based matching (no LLM)",
        }

    def _empty_result(self, analysis: str) -> dict:
        """返回空结果"""
        return {
            "api_requests": [],
            "example_requests": [],
            "skipped": [],
            "analysis": analysis,
        }
