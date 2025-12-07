# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Kernel + Tiling 联合分支"""

from typing import TYPE_CHECKING

from ..parsers import parse_code, parse_json

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
        执行 Joint 分支

        Args:
            python_ref: Python 参考实现代码
        """
        self._verbose("[Joint] Starting...")

        # Phase 1: 联合规划讨论
        self._verbose("[Joint] Phase 1: Joint Planning...")
        self._joint_planning(python_ref)

        # Phase 2: 知识检索
        self._verbose("[Joint] Phase 2: Knowledge Retrieval...")
        self._retrieve_knowledge()

        # Phase 3: 代码实现
        self._verbose("[Joint] Phase 3: Code Implementation...")
        self._implement_code(python_ref)

        self._verbose("[Joint] Done")

    def _joint_planning(self, python_ref: str):
        """多轮对话达成共识"""
        context = {
            "signature": self.run_state_dict.signature,
            "compute_pattern": self.run_state_dict.compute_pattern,
            "python_ref": python_ref
        }
        conversation = []

        for turn in range(self.config.max_joint_turns):
            self._verbose(f"[Joint] Turn {turn + 1}")

            # Tiling 专员提出策略
            tiling_prompt = self.config.interface.get_tiling_propose_prompt(context, conversation)
            tiling_msg, _ = self.config.running_llm.get_response(tiling_prompt)
            conversation.append({"role": "tiling", "content": tiling_msg})

            # Kernel 专员评审
            kernel_prompt = self.config.interface.get_kernel_review_prompt(context, conversation)
            kernel_msg, _ = self.config.running_llm.get_response(kernel_prompt)
            conversation.append({"role": "kernel", "content": kernel_msg})

            # 检查是否达成共识
            if self._check_consensus(kernel_msg):
                self._verbose("[Joint] Consensus reached")
                break

        self.run_state_dict.joint_conversation = conversation
        self.run_state_dict.joint_plan = self._extract_joint_plan(conversation)

    def _check_consensus(self, kernel_msg: str) -> bool:
        """检查 Kernel 专员是否接受方案"""
        return "accepted" in kernel_msg.lower() or "agree" in kernel_msg.lower()

    def _extract_joint_plan(self, conversation: list) -> dict:
        """从对话中提取联合规划"""
        # TODO: 实现从对话中提取结构化规划
        return {
            "conversation": conversation,
            "retrieval_requests": []
        }

    def _retrieve_knowledge(self):
        """根据规划检索知识"""
        if not self.config.knowledge_base:
            self.run_state_dict.knowledge = {}
            return

        knowledge = {}
        retrieval_requests = self.run_state_dict.joint_plan.get("retrieval_requests", [])

        for req in retrieval_requests:
            if req.get("type") == "api":
                api_name = req.get("name")
                knowledge[f"api_{api_name}"] = self.config.knowledge_base.search_api(api_name)
            elif req.get("type") == "example":
                op_name = req.get("name")
                knowledge[f"example_{op_name}"] = self.config.knowledge_base.search_operator(op_name)

        self.run_state_dict.knowledge = knowledge

    def _implement_code(self, python_ref: str):
        """代码实现"""
        # Kernel 必须生成
        kernel_prompt = self.config.interface.get_kernel_impl_prompt(
            self.run_state_dict.joint_plan,
            self.run_state_dict.knowledge,
            python_ref
        )
        response, _ = self.config.running_llm.get_response(kernel_prompt)
        self.run_state_dict.kernel_src = parse_code(response)

        # Tiling 根据策略决定
        if self.run_state_dict.strategies.get("tiling") == "default":
            self.run_state_dict.tiling_src = None
            self.run_state_dict.operator_src = None
        else:
            tiling_prompt = self.config.interface.get_tiling_impl_prompt(
                self.run_state_dict.joint_plan,
                self.run_state_dict.knowledge
            )
            response, _ = self.config.running_llm.get_response(tiling_prompt)
            result = parse_json(response)
            self.run_state_dict.tiling_src = result.get("host_tiling_src")
            self.run_state_dict.operator_src = result.get("host_operator_src")
