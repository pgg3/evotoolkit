# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Debug Loop: 迭代调试循环"""

from typing import TYPE_CHECKING

from ..parsers import parse_code, parse_json

if TYPE_CHECKING:
    from ..run_config import CANNIniterConfig
    from ..run_state_dict import CANNIniterRunStateDict


class DebugLoop:
    """迭代调试循环"""

    def __init__(self, config: "CANNIniterConfig", run_state_dict: "CANNIniterRunStateDict"):
        self.config = config
        self.run_state_dict = run_state_dict

    def _verbose(self, msg: str):
        """输出信息"""
        if self.config.verbose:
            print(msg)

    def run(self, python_ref: str) -> dict:
        """
        执行调试循环

        Args:
            python_ref: Python 参考实现代码

        Returns:
            {"success": bool, "code": dict}
        """
        for iteration in range(self.config.max_debug_iterations):
            self.run_state_dict.current_iteration = iteration
            self._verbose(f"[Debug] Iteration {iteration + 1}/{self.config.max_debug_iterations}")

            # 组装代码
            full_code = self._assemble_code()

            # 评估
            result = self.config.task.evaluate_code(full_code)

            if result.valid:
                self._verbose("[Debug] SUCCESS!")
                self.run_state_dict.success = True
                return {"success": True, "code": full_code}

            # 记录错误
            error_info = result.additional_info or {}
            self._verbose(f"[Debug] Error: {error_info.get('stage', 'unknown')}")

            # 错误分类 + 分派修复
            error_type = self._classify_error(error_info)
            self._dispatch_fix(error_type, error_info)

            self.run_state_dict.debug_history.append({
                "iteration": iteration,
                "error_type": error_type,
                "error": error_info
            })

        self._verbose("[Debug] Max iterations reached")
        return {"success": False, "code": self._assemble_code()}

    def _assemble_code(self) -> dict:
        """组装完整代码"""
        return {
            "kernel_src": self.run_state_dict.kernel_src,
            "host_tiling_src": self.run_state_dict.tiling_src,
            "host_operator_src": self.run_state_dict.operator_src,
            "python_bind_src": self.run_state_dict.pybind_src,
        }

    def _classify_error(self, error_info: dict) -> str:
        """错误分类"""
        stage = error_info.get("stage", "")
        error_msg = str(error_info.get("error", ""))

        if stage == "compile":
            if "kernel" in error_msg.lower() or ".cpp" in error_msg:
                return "kernel"
            elif "tiling" in error_msg.lower() or "host" in error_msg.lower():
                return "tiling"
            elif "pybind" in error_msg.lower() or "bind" in error_msg.lower():
                return "pybind"
        elif stage == "correctness":
            return "kernel"
        elif stage == "deploy":
            return "pybind"

        return "unknown"

    def _dispatch_fix(self, error_type: str, error_info: dict):
        """分派给对应专员修复"""
        self._verbose(f"[Debug] Dispatching to {error_type} agent...")

        if error_type == "kernel":
            self._fix_kernel(error_info)
        elif error_type == "tiling":
            self._fix_tiling(error_info)
        elif error_type == "pybind":
            self._fix_pybind(error_info)
        else:
            # unknown: 尝试修复 kernel
            self._verbose("[Debug] Unknown error, attempting kernel fix...")
            self._fix_kernel(error_info)

    def _fix_kernel(self, error_info: dict):
        """修复 Kernel 代码"""
        prompt = self.config.interface.get_debug_prompt(
            "kernel", self.run_state_dict.kernel_src, error_info
        )
        response, _ = self.config.running_llm.get_response(prompt)
        self.run_state_dict.kernel_src = parse_code(response)

    def _fix_tiling(self, error_info: dict):
        """修复 Tiling 代码"""
        prompt = self.config.interface.get_debug_prompt(
            "tiling",
            {"tiling": self.run_state_dict.tiling_src, "operator": self.run_state_dict.operator_src},
            error_info
        )
        response, _ = self.config.running_llm.get_response(prompt)
        result = parse_json(response)
        self.run_state_dict.tiling_src = result.get("host_tiling_src")
        self.run_state_dict.operator_src = result.get("host_operator_src")

    def _fix_pybind(self, error_info: dict):
        """修复 Pybind 代码"""
        prompt = self.config.interface.get_debug_prompt(
            "pybind", self.run_state_dict.pybind_src, error_info
        )
        response, _ = self.config.running_llm.get_response(prompt)
        self.run_state_dict.pybind_src = parse_code(response)
