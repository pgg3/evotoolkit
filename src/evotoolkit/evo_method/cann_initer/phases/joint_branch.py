# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Kernel + Tiling 联合分支"""

import re
from typing import TYPE_CHECKING, List

from ..parsers import parse_code, parse_tag, parse_multiple_tags

if TYPE_CHECKING:
    from ..run_config import CANNIniterConfig
    from ..run_state_dict import CANNIniterRunStateDict


class JointBranch:
    """Kernel + Tiling 联合分支（多轮对话达成共识）"""

    def __init__(self, config: "CANNIniterConfig", run_state_dict: "CANNIniterRunStateDict"):
        self.config = config
        self.run_state_dict = run_state_dict

    def _verbose(self, msg: str):
        """输出信息"""
        if self.config.verbose:
            print(msg)

    def run(self, python_ref: str):
        """
        执行 Joint 分支完整流程 (Phase 1-3)

        Args:
            python_ref: Python 参考实现代码
        """
        self._verbose("[Joint] Starting...")

        # Phase 1 + 2: Planning
        self.run_planning(python_ref)

        # Phase 3: 代码实现
        self._verbose("[Joint] Phase 3: Code Implementation...")
        self._implement_code(python_ref)

        self._verbose("[Joint] Done")

    def run_planning(self, python_ref: str):
        """
        仅执行 Planning 阶段 (Phase 1 + Phase 2)

        输出:
            - run_state_dict.joint_plan
            - run_state_dict.knowledge_context

        Args:
            python_ref: Python 参考实现代码
        """
        # Phase 1: 联合规划讨论
        self._verbose("[Joint] Phase 1: Joint Planning...")
        self._joint_planning(python_ref)

        # Phase 2: 知识检索
        self._verbose("[Joint] Phase 2: Knowledge Retrieval...")
        self._retrieve_knowledge()

    def run_implementation(self, python_ref: str):
        """
        仅执行 Implementation 阶段 (Phase 3)

        前置条件:
            - run_state_dict.joint_plan 已填充
            - run_state_dict.knowledge_context 已填充 (可选)

        Args:
            python_ref: Python 参考实现代码
        """
        self._verbose("[Joint] Phase 3: Code Implementation...")
        self._implement_code(python_ref)
        self._verbose("[Joint] Done")

    def _joint_planning(self, python_ref: str):
        """多轮对话达成共识

        Tiling Agent 和 Kernel Agent 通过多轮对话确定分块策略和 kernel 设计。
        如果 Tiling Agent 在 prompt 层面判断适合使用 default 策略，
        可通过流控提前结束对话。
        """
        context = {
            "signature": self.run_state_dict.signature,
            "python_ref": python_ref,
            "npu_type": self.config.task.npu_type,
        }
        conversation = []

        # 多轮对话
        for turn in range(self.config.max_joint_turns):
            self._verbose(f"[Joint] Turn {turn + 1}")
            is_final_round = (turn == self.config.max_joint_turns - 1)

            # Tiling 专员提出策略
            tiling_prompt = self.config.interface.get_tiling_propose_prompt(context, conversation)
            tiling_msg, _ = self.config.running_llm.get_response(tiling_prompt)
            conversation.append({"role": "tiling", "content": tiling_msg})

            # Kernel 专员评审（最后一轮强制要求输出可实现的方案）
            kernel_prompt = self.config.interface.get_kernel_review_prompt(
                context, conversation, is_final_round=is_final_round
            )
            kernel_msg, _ = self.config.running_llm.get_response(kernel_prompt)
            conversation.append({"role": "kernel", "content": kernel_msg})

            # 检查是否达成共识
            if self._check_consensus(kernel_msg):
                self._verbose("[Joint] Consensus reached")
                break
            elif is_final_round:
                self._verbose("[Joint] Final round - forcing consensus for implementation")

        self.run_state_dict.joint_conversation = conversation
        self.run_state_dict.joint_plan = self._extract_joint_plan(conversation)

    def _check_consensus(self, kernel_msg: str) -> bool:
        """检查 Kernel 专员是否接受方案

        解析 <response> 块中的 accepted 字段：
        - accepted: true -> 达成共识
        - accepted: false -> 需要继续修订
        """
        # 提取 <response> 块
        start = kernel_msg.find('<response>')
        end = kernel_msg.find('</response>')

        if start == -1 or end == -1:
            # 没有找到标准格式，尝试直接解析
            content = kernel_msg
        else:
            content = kernel_msg[start + len('<response>'):end]

        # 解析 accepted 字段（忽略大小写和空白）
        content_lower = content.lower()

        # 检查 "accepted: true" 或 "accepted:true"
        if 'accepted: true' in content_lower or 'accepted:true' in content_lower:
            return True

        # 检查 "accepted: false" 或 "accepted:false"
        if 'accepted: false' in content_lower or 'accepted:false' in content_lower:
            return False

        # 未找到明确的 accepted 字段，默认为未达成共识
        return False

    def _extract_joint_plan(self, conversation: list) -> dict:
        """从对话中提取联合规划

        提取最后两条消息：
        - 倒数第二条（tiling agent）：tiling 策略
        - 最后一条（kernel agent）：kernel 设计 + Tiling Fields + Useful References

        策略判断优先级：
        1. 优先从 Kernel Agent 输出中解析 `strategy: default/generate`
        2. 如果没有，根据是否有 Tiling Fields Required 判断
        3. 最后回退到 Tiling Agent 的策略

        Returns:
            dict: {
                "tiling_proposal": str,  # tiling agent 的 <response> 内容
                "kernel_design": str,    # kernel agent 的 <response> 内容
                "tiling_strategy": str,  # "default" or "generate"
                "tiling_fields": list,   # 从 kernel design 中提取 (仅 generate 策略)
                "kernel_pseudocode": str,  # 提取的 kernel 伪代码
                "tiling_execution": str,   # 提取的 tiling 执行伪代码
                "retrieval_requests": [{"type": "api"|"example", "name": str}, ...]
            }
        """
        if len(conversation) < 2:
            return {"tiling_proposal": None, "kernel_design": None,
                    "tiling_strategy": "default", "tiling_fields": [],
                    "kernel_pseudocode": None, "tiling_execution": None,
                    "retrieval_requests": []}

        # 倒数第二条是 tiling agent，最后一条是 kernel agent
        tiling_msg = conversation[-2] if conversation[-2].get('role') == 'tiling' else None
        kernel_msg = conversation[-1] if conversation[-1].get('role') == 'kernel' else None

        # 提取 <response> 块内容
        tiling_proposal = self._extract_response_block(
            tiling_msg.get('content', '')) if tiling_msg else None
        kernel_design = self._extract_response_block(
            kernel_msg.get('content', '')) if kernel_msg else None

        # 判断 kernel 是否 accepted
        kernel_accepted = kernel_design and (
            'accepted: true' in kernel_design.lower() or
            'accepted:true' in kernel_design.lower()
        )

        # 判断是否是 final round（通过检查 kernel 输出是否有 ## Tiling Execution）
        is_final_round = kernel_design and '## tiling execution' in kernel_design.lower()

        # 解析 tiling 策略（优先从 Kernel Agent 输出）
        tiling_strategy = self._parse_kernel_strategy(kernel_design) if kernel_design else None
        if not tiling_strategy:
            # 回退：从 Tiling Agent 输出解析
            tiling_strategy = self._parse_tiling_strategy(tiling_proposal)

        # 根据策略提取 tiling fields
        tiling_fields = []
        if tiling_strategy == "generate":
            # generate 策略：从 kernel design 提取 Tiling Fields Required
            if kernel_accepted and kernel_design:
                tiling_fields = self._parse_tiling_fields(kernel_design)
            # 如果没有，尝试从 tiling proposal 提取
            if not tiling_fields and tiling_proposal:
                tiling_fields = self._parse_tiling_fields(tiling_proposal)
        # default 策略：不需要自定义 tiling fields

        # 提取 kernel pseudocode (仅当 kernel accepted 时有效)
        kernel_pseudocode = None
        if kernel_accepted and kernel_design:
            kernel_pseudocode = self._parse_kernel_pseudocode(kernel_design)

        # 提取 tiling execution / execution flow
        tiling_execution = None
        if is_final_round:
            # Final round: ## Tiling Execution 在 kernel output
            tiling_execution = self._parse_tiling_execution(kernel_design)
        elif tiling_strategy == "generate":
            # 非 final + generate: ## Execution Flow 在 tiling agent output
            tiling_execution = self._parse_execution_flow(tiling_proposal) if tiling_proposal else None
        # default 策略：不需要 execution flow

        # 更新 run_state_dict.strategies
        if not hasattr(self.run_state_dict, 'strategies') or self.run_state_dict.strategies is None:
            self.run_state_dict.strategies = {}
        self.run_state_dict.strategies["tiling"] = tiling_strategy

        # 解析 Useful References 生成 retrieval_requests (仅当 kernel accepted 时有效)
        retrieval_requests = []
        if kernel_accepted and kernel_design:
            retrieval_requests = self._parse_useful_references(kernel_design)

        return {
            "tiling_proposal": tiling_proposal,
            "kernel_design": kernel_design,
            "tiling_strategy": tiling_strategy,
            "tiling_fields": tiling_fields,
            "kernel_pseudocode": kernel_pseudocode,
            "tiling_execution": tiling_execution,
            "retrieval_requests": retrieval_requests,
        }

    def _extract_response_block(self, content: str) -> str:
        """提取 <response> 块的内容"""
        start = content.find('<response>')
        end = content.find('</response>')
        if start != -1 and end != -1:
            return content[start + len('<response>'):end].strip()
        return content.strip()

    def _parse_tiling_strategy(self, tiling_proposal: str) -> str:
        """解析 tiling 策略类型

        从 tiling proposal 中解析 "Strategy: default" 或 "Strategy: generate"
        """
        if not tiling_proposal:
            return "default"

        proposal_lower = tiling_proposal.lower()
        if 'strategy: default' in proposal_lower or 'strategy:default' in proposal_lower:
            return "default"
        if 'strategy: generate' in proposal_lower or 'strategy:generate' in proposal_lower:
            return "generate"

        # 未找到明确策略，默认为 generate（更安全）
        return "generate"

    def _parse_kernel_strategy(self, kernel_design: str) -> str:
        """解析 Kernel Agent 输出中的策略

        从 kernel design 中解析 "strategy: default" 或 "strategy: generate"

        Returns:
            str: "default" 或 "generate"，如果未找到则返回 None
        """
        if not kernel_design:
            return None

        design_lower = kernel_design.lower()

        # 查找 "strategy: default" 或 "strategy:default"
        if 'strategy: default' in design_lower or 'strategy:default' in design_lower:
            return "default"

        # 查找 "strategy: generate" 或 "strategy:generate"
        if 'strategy: generate' in design_lower or 'strategy:generate' in design_lower:
            return "generate"

        # 未找到明确策略
        return None

    def _parse_tiling_fields(self, kernel_design: str) -> List[dict]:
        """解析 Tiling Fields Required 部分

        格式（来自 kernel agent final round）：
        ## Tiling Fields Required
        - batchSize: uint32_t // B
        - seqLen: uint32_t // sequence length
        - dModel: uint32_t // model dimension

        Returns:
            list: [{"name": str, "type": str, "purpose": str}, ...]
        """
        fields = []
        if not kernel_design:
            return fields

        # 找到 Tiling Fields Required 部分
        fields_start = kernel_design.lower().find('## tiling fields required')
        if fields_start == -1:
            # 也尝试匹配 "## Tiling Fields"
            fields_start = kernel_design.lower().find('## tiling fields')
            if fields_start == -1:
                return fields

        # 从该位置开始提取，直到下一个 ## 或结束
        rest = kernel_design[fields_start:]
        # 找到下一个 section（## 开头）
        next_section = rest.find('\n##', 1)
        if next_section != -1:
            fields_section = rest[:next_section]
        else:
            fields_section = rest

        # 解析每一行 "- name: type // purpose" 或 "- name: type"
        for line in fields_section.split('\n'):
            line = line.strip()
            if not line.startswith('-'):
                continue

            line = line[1:].strip()  # 去掉 "-"

            # 解析 "name: type // purpose" 格式
            if '//' in line:
                main_part, purpose = line.split('//', 1)
                purpose = purpose.strip()
            else:
                main_part = line
                purpose = ""

            if ':' in main_part:
                name, type_str = main_part.split(':', 1)
                name = name.strip()
                type_str = type_str.strip()
                if name:
                    fields.append({
                        "name": name,
                        "type": type_str,
                        "purpose": purpose
                    })

        return fields

    def _parse_kernel_pseudocode(self, content: str) -> str:
        """解析 Kernel Pseudocode 部分

        格式：
        ## Kernel Pseudocode
        ```cpp
        // code here
        ```
        """
        if not content:
            return None

        # 找到 ## Kernel Pseudocode 部分
        section_start = content.lower().find('## kernel pseudocode')
        if section_start == -1:
            return None

        rest = content[section_start:]

        # 找到代码块
        code_start = rest.find('```')
        if code_start == -1:
            return None

        # 跳过 ```cpp 或 ``` 行
        code_start = rest.find('\n', code_start) + 1
        code_end = rest.find('```', code_start)
        if code_end == -1:
            return None

        return rest[code_start:code_end].strip()

    def _parse_tiling_execution(self, content: str) -> str:
        """解析 Kernel Final Round 的 Tiling Execution 部分

        格式 (from kernel_prompts.py final round):
        ## Tiling Execution
        ```
        for <loop>:
            CopyIn: <data>
            Compute: <ops>
            CopyOut: <data>
        ```
        """
        if not content:
            return None

        content_lower = content.lower()

        # 只查找 ## Tiling Execution (Kernel Final Round 特有)
        section_start = content_lower.find('## tiling execution')
        if section_start == -1:
            return None

        rest = content[section_start:]

        # 找到代码块
        code_start = rest.find('```')
        if code_start == -1:
            return None

        # 跳过 ``` 行
        code_start = rest.find('\n', code_start) + 1
        code_end = rest.find('```', code_start)
        if code_end == -1:
            return None

        return rest[code_start:code_end].strip()

    def _parse_execution_flow(self, content: str) -> str:
        """解析 Tiling Agent 的 Execution Flow 部分

        格式 (from tiling_prompts.py):
        ## Execution Flow
        ```
        for tile in range(tilesPerCore):
            CopyIn: x[tile_start : tile_end], y[tile_start : tile_end]
            CopyOut: z[tile_start : tile_end]
        ```
        Note: Element-wise, tiles are independent.
        """
        if not content:
            return None

        content_lower = content.lower()

        # 查找 ## Execution Flow (Tiling Agent 输出)
        section_start = content_lower.find('## execution flow')
        if section_start == -1:
            return None

        rest = content[section_start:]

        # 找到代码块
        code_start = rest.find('```')
        if code_start == -1:
            return None

        # 跳过 ``` 行
        code_start = rest.find('\n', code_start) + 1
        code_end = rest.find('```', code_start)
        if code_end == -1:
            return None

        return rest[code_start:code_end].strip()

    def _parse_useful_references(self, kernel_design: str) -> List[dict]:
        """解析 Useful References 部分，生成 retrieval_requests

        格式：
        ## Useful References
        - APIs: [API1, API2 (Desc), ...]
        - Examples: [example1, example2, ...]
        """
        requests = []

        # 找到 Useful References 部分
        ref_start = kernel_design.lower().find('## useful references')
        if ref_start == -1:
            return requests

        ref_section = kernel_design[ref_start:]

        # 解析 APIs: [...]
        apis_match = re.search(r'-\s*APIs?:\s*\[([^\]]*)\]', ref_section, re.IGNORECASE)
        if apis_match:
            apis_str = apis_match.group(1)
            for api in apis_str.split(','):
                api = api.strip()
                if not api:
                    continue
                # 清理括号内的说明，如 "MatMul (Cube)" -> "MatMul"
                api_name = re.sub(r'\s*\([^)]*\)', '', api).strip()
                if api_name:
                    requests.append({"type": "api", "name": api_name})

        # 解析 Examples: [...]
        examples_match = re.search(r'-\s*Examples?:\s*\[([^\]]*)\]', ref_section, re.IGNORECASE)
        if examples_match:
            examples_str = examples_match.group(1)
            for example in examples_str.split(','):
                example = example.strip()
                if example:
                    requests.append({"type": "example", "name": example})

        return requests

    def _retrieve_knowledge(self):
        """根据规划检索知识

        Two-stage retrieval:
        1. RetrievalPlanner: 概念性请求 → 精确请求
        2. KnowledgeSummarizer: 原始知识 → 精简摘要
        """
        if not self.config.knowledge_base:
            self.run_state_dict.knowledge = {}
            self.run_state_dict.knowledge_context = ""
            return

        joint_plan = self.run_state_dict.joint_plan
        raw_requests = joint_plan.get("retrieval_requests", [])

        # Create LLM callback if available
        llm_call = None
        if self.config.running_llm:
            def llm_call(prompt: str) -> str:
                response, _ = self.config.running_llm.get_response(prompt)
                return response

        # Stage 1: RetrievalPlanner (概念性请求 → 精确请求)
        from ..knowledge import RetrievalPlanner, KnowledgeSummarizer

        planner = RetrievalPlanner(self.config.knowledge_base, llm_client=llm_call)
        plan_result = planner.plan(
            operator_signature=self.run_state_dict.signature,
            kernel_pseudocode=joint_plan.get("kernel_pseudocode", ""),
            tiling_execution=joint_plan.get("tiling_execution", ""),
            tiling_fields=joint_plan.get("tiling_fields", []),
            raw_requests=raw_requests,
        )

        self._verbose(f"[Joint] RetrievalPlanner: {len(plan_result.get('api_requests', []))} APIs, "
                     f"{len(plan_result.get('example_requests', []))} examples")

        # Fetch raw knowledge
        raw_knowledge = {"apis": {}, "examples": {}}
        kb = self.config.knowledge_base

        for req in plan_result.get("api_requests", []):
            name = req.get("name")
            if name:
                raw_knowledge["apis"][name] = kb.search_api(name)

        for req in plan_result.get("example_requests", []):
            name = req.get("name")
            if name:
                raw_knowledge["examples"][name] = kb.search_operator(name)

        # Stage 2: KnowledgeSummarizer (原始知识 → 精简摘要)
        cann_path = getattr(self.config.knowledge_base.config, 'cann_path', None) \
            if hasattr(self.config.knowledge_base, 'config') else None

        summarizer = KnowledgeSummarizer(
            llm_client=llm_call,
            max_examples=2,
            cann_path=cann_path,
        )

        summarized = summarizer.summarize(
            task_context={
                "operator_signature": self.run_state_dict.signature,
                "kernel_pseudocode": joint_plan.get("kernel_pseudocode", ""),
                "tiling_execution": joint_plan.get("tiling_execution", ""),
                "tiling_fields": joint_plan.get("tiling_fields", []),
            },
            raw_knowledge=raw_knowledge,
        )

        self._verbose(f"[Joint] KnowledgeSummarizer: {len(summarized.get('api_summaries', []))} API summaries, "
                     f"{len(summarized.get('example_summaries', []))} example summaries")

        # Store results
        self.run_state_dict.knowledge = raw_knowledge
        self.run_state_dict.knowledge_summary = summarized  # Full result: api_summaries, example_summaries, combined_context
        self.run_state_dict.knowledge_context = summarized.get("combined_context", "")
        self.run_state_dict.retrieval_plan = plan_result

    def _get_infer_shape_body(self) -> str:
        """获取 InferShape body（从 pybind 的 shape_inference_code 翻译）

        逻辑：
        1. 如果 strategies 为 default：返回空（使用默认模板）
        2. 如果有 shape_inference_code：翻译成 Ascend 格式
        3. 如果没有：返回空（使用默认模板）
        """
        # Default strategy case: use default template in assemble_tiling_host
        if self.run_state_dict.strategies.get("tiling") == "default":
            self._verbose("[Joint] InferShape: using default (tiling strategy = default)")
            return ""

        # Get shape_inference_code from pybind branch
        shape_code = getattr(self.run_state_dict, 'shape_inference_code', None)
        if not shape_code:
            self._verbose("[Joint] InferShape: no shape_inference_code, using default")
            return ""

        # Translate using InferShapeTranslator
        from .infer_shape_translator import InferShapeTranslator

        # Create LLM callback
        llm_call = None
        if self.config.running_llm:
            def llm_call(prompt: str) -> str:
                response, _ = self.config.running_llm.get_response(prompt)
                return response

        translator = InferShapeTranslator(llm_client=llm_call)
        signature = self.run_state_dict.signature
        sig_dict = signature.to_dict() if hasattr(signature, 'to_dict') else (signature or {})

        # Check if tiling uses default strategy (implies output = input shape)
        use_default = self.run_state_dict.strategies.get("tiling") == "default"

        infer_shape_body = translator.translate(
            shape_inference_code=shape_code,
            signature=sig_dict,
            output_equals_input_shape=use_default,
        )
        self._verbose(f"[Joint] InferShape: translated from pybind shape code")
        return infer_shape_body

    def _implement_code(self, python_ref: str):
        """三阶段代码实现 (使用 assemble 模式)

        新设计模式：
        - Prompt 只要求 LLM 生成变量部分
        - 使用 assemble_* 方法组装完整代码
        - 固定结构由程序控制，防止 LLM 生成错误

        Stage 1: tiling.h (仅 generate tiling)
        Stage 2: op_host.cpp (仅 generate tiling)
        Stage 3: op_kernel.cpp (总是生成)

        使用 knowledge_context（KnowledgeSummarizer 的输出）作为知识上下文
        """
        # 准备 context
        # NOTE: python_ref is included in context for completeness, but the prompt
        # functions intentionally DO NOT use it. The pseudocode/execution from
        # Joint Plan already contains the translated logic, making python_ref
        # redundant and potentially confusing for the LLM.
        context = {
            "op_name": self.run_state_dict.op_name,
            "signature": self.run_state_dict.signature,
            "python_ref": python_ref,  # Available but not used by prompts
            "shape_inference": self.run_state_dict.shape_inference,
            "strategies": self.run_state_dict.strategies,
        }
        op_name = self.run_state_dict.op_name
        knowledge = self.run_state_dict.knowledge_context
        plan = self.run_state_dict.joint_plan
        tiling_header = None

        # Tiling 根据策略决定
        if self.run_state_dict.strategies.get("tiling") == "default":
            # Default tiling: 不需要生成 tiling.h 和 op_host.cpp
            self.run_state_dict.tiling_src = None
            self.run_state_dict.operator_src = None
        else:
            # Custom tiling: 三阶段生成

            # Stage 1: 生成 tiling.h
            self._verbose("[Joint] Stage 1: Generating tiling.h...")
            tiling_header_prompt = self.config.interface.get_tiling_header_prompt(
                plan, context
            )
            response, _ = self.config.running_llm.get_response(tiling_header_prompt)

            # 解析 <response> 标签，组装完整代码
            field_definitions = parse_tag(response, "response")
            if not field_definitions:
                # 回退：尝试解析代码块
                field_definitions = parse_code(response)

            tiling_header = self.config.interface.assemble_tiling_header(
                op_name, field_definitions
            )
            self.run_state_dict.tiling_src = tiling_header

            # Stage 2: 生成 op_host.cpp
            self._verbose("[Joint] Stage 2: Generating op_host.cpp...")
            tiling_host_prompt = self.config.interface.get_tiling_host_prompt(
                plan, context, knowledge, tiling_header
            )
            response, _ = self.config.running_llm.get_response(tiling_host_prompt)

            # 解析标签，组装完整代码
            tiling_func_body = parse_tag(response, "tiling_func_body")
            input_output_defs = parse_tag(response, "input_output_defs")

            # InferShape: 从 pybind 的 shape_inference_code 翻译
            infer_shape_body = self._get_infer_shape_body()

            if not tiling_func_body:
                # 回退：尝试解析代码块
                self._verbose("[Joint] Warning: Failed to parse tiling_func_body tag, falling back to code block")
                self.run_state_dict.operator_src = parse_code(response)
            else:
                self.run_state_dict.operator_src = self.config.interface.assemble_tiling_host(
                    op_name, tiling_func_body, input_output_defs, infer_shape_body
                )

        # Stage 3: 生成 op_kernel.cpp (总是需要)
        self._verbose("[Joint] Stage 3: Generating op_kernel.cpp...")
        kernel_prompt = self.config.interface.get_kernel_impl_prompt(
            plan, context, knowledge, tiling_header
        )
        response, _ = self.config.running_llm.get_response(kernel_prompt)

        # 解析 7 个标签 (flexible structure)
        kernel_tags = [
            "init_params", "init_body", "process_body",
            "private_methods", "member_vars",
            "global_func_params", "init_call_args"
        ]
        parsed = parse_multiple_tags(response, kernel_tags)

        # 检查是否成功解析
        if parsed["init_params"] and parsed["init_body"]:
            self.run_state_dict.kernel_src = self.config.interface.assemble_kernel_impl(
                op_name,
                init_params=parsed["init_params"],
                init_body=parsed["init_body"],
                process_body=parsed["process_body"],
                private_methods=parsed["private_methods"],
                member_vars=parsed["member_vars"],
                global_func_params=parsed["global_func_params"],
                init_call_args=parsed["init_call_args"],
            )
        else:
            # 回退：尝试解析代码块
            self._verbose("[Joint] Warning: Failed to parse kernel tags, falling back to code block")
            self.run_state_dict.kernel_src = parse_code(response)
