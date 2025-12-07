# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Pybind 独立分支"""

from typing import TYPE_CHECKING

from ..parsers import parse_code

if TYPE_CHECKING:
    from ..run_config import CANNIniterConfig
    from ..run_state_dict import CANNIniterRunStateDict


class PybindBranch:
    """Pybind 独立分支（简单上下文，可独立并行）"""

    def __init__(self, config: "CANNIniterConfig", run_state_dict: "CANNIniterRunStateDict"):
        self.config = config
        self.run_state_dict = run_state_dict

    def _verbose(self, msg: str):
        """输出信息"""
        if self.config.verbose:
            print(msg)

    def run(self):
        """执行 Pybind 分支"""
        self._verbose("[Pybind] Starting...")

        if self.run_state_dict.strategies.get("pybind") == "default":
            self._verbose("[Pybind] Using default template")
            self.run_state_dict.pybind_src = None
        else:
            self._verbose("[Pybind] Generating with LLM...")
            prompt = self.config.interface.get_pybind_prompt(
                self.run_state_dict.signature
            )
            response, _ = self.config.running_llm.get_response(prompt)
            self.run_state_dict.pybind_src = parse_code(response)

        self._verbose("[Pybind] Done")
