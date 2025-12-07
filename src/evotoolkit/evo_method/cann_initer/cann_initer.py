# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANNIniter: Ascend C 算子自动生成 Agent

基于 Tool-based Retrieval + 专员分工协作的设计理念：
- Phase 0: 签名解析（确定性）+ 计算模式识别（LLM）
- 并行分支: Pybind 独立 || Kernel+Tiling 联合（多轮对话）
- Debug Loop: 迭代调试直到正确
"""

import concurrent.futures
import os
from pathlib import Path

from .phases import DebugLoop, JointBranch, Phase0Analyzer, PybindBranch
from .run_config import CANNIniterConfig
from .run_state_dict import CANNIniterRunStateDict


class CANNIniter:
    """CANNIniter 主流程"""

    def __init__(self, config: CANNIniterConfig):
        self.config = config
        self.run_state_dict = self._load_or_create_state()

    def _load_or_create_state(self) -> CANNIniterRunStateDict:
        """加载或创建状态"""
        state_file = os.path.join(self.config.output_path, "run_state.pkl")
        if os.path.exists(state_file):
            self._verbose("Loading state from pickle...")
            return CANNIniterRunStateDict.from_pickle(state_file)
        return CANNIniterRunStateDict()

    def _save_state(self):
        """保存状态"""
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        state_file = os.path.join(self.config.output_path, "run_state.pkl")
        self.run_state_dict.to_pickle(state_file)

    def _verbose(self, msg: str):
        """输出信息"""
        if self.config.verbose:
            print(msg)

    def run(self, op_name: str, python_ref: str) -> dict:
        """
        执行完整的算子生成流程

        Args:
            op_name: 算子名称
            python_ref: Python 参考实现代码

        Returns:
            {"success": bool, "code": dict}
        """
        self._verbose(f"\n{'='*60}")
        self._verbose(f"CANNIniter: {op_name}".center(60))
        self._verbose("=" * 60)

        # Phase 0: 签名解析 + 计算模式识别
        self._verbose("\n--- Phase 0: Signature Analysis ---")
        phase0 = Phase0Analyzer(self.config, self.run_state_dict)
        phase0.analyze(op_name, python_ref)
        self._save_state()

        # 并行分支处理
        self._verbose("\n--- Parallel Branches: Pybind || Joint ---")
        pybind_branch = PybindBranch(self.config, self.run_state_dict)
        joint_branch = JointBranch(self.config, self.run_state_dict)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pybind_future = executor.submit(pybind_branch.run)
            joint_future = executor.submit(joint_branch.run, python_ref)
            pybind_future.result()
            joint_future.result()
        self._save_state()

        # Evaluate + Debug Loop
        self._verbose("\n--- Debug Loop ---")
        debug_loop = DebugLoop(self.config, self.run_state_dict)
        result = debug_loop.run(python_ref)

        self.run_state_dict.is_done = True
        self._save_state()

        return result
