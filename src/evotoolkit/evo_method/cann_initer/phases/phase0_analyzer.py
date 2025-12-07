# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Phase 0: 签名解析 + 计算模式识别"""

from typing import TYPE_CHECKING

from ..parsers import parse_json

if TYPE_CHECKING:
    from ..run_config import CANNIniterConfig
    from ..run_state_dict import CANNIniterRunStateDict


class Phase0Analyzer:
    """Phase 0: 签名解析（确定性）+ 计算模式识别（LLM）"""

    def __init__(self, config: "CANNIniterConfig", run_state_dict: "CANNIniterRunStateDict"):
        self.config = config
        self.run_state_dict = run_state_dict

    def _verbose(self, msg: str):
        """输出信息"""
        if self.config.verbose:
            print(msg)

    def analyze(self, op_name: str, python_ref: str):
        """
        执行 Phase 0 分析

        Args:
            op_name: 算子名称
            python_ref: Python 参考实现代码
        """
        # 1. 签名解析（复用 Evaluator 的 parser）
        self._verbose("Parsing signature...")
        self.run_state_dict.signature = self.config.task.parser.parse(python_ref, op_name)

        # 2. 计算模式识别（LLM）
        self._verbose("Analyzing compute pattern with LLM...")
        prompt = self.config.interface.get_pattern_analysis_prompt(
            python_ref, self.run_state_dict.signature
        )
        response, _ = self.config.running_llm.get_response(prompt)
        result = parse_json(response)

        self.run_state_dict.compute_pattern = result.get("compute_pattern", "other")
        self.run_state_dict.strategies = result.get("strategies", {
            "kernel": "generate",
            "tiling": "generate",
            "pybind": "generate"
        })

        self._verbose(f"Compute pattern: {self.run_state_dict.compute_pattern}")
        self._verbose(f"Strategies: {self.run_state_dict.strategies}")
