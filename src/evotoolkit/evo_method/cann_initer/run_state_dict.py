# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License
# 位置： evotoolkit-master\src\evotoolkit\evo_method\cann_initer
"""
CANNIniter run state for inter-phase data passing.
添加了 latest_analysis
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


class CANNIniterRunStateDict:
    """CANNIniter inter-phase state (serialized with pickle)."""

    def __init__(
        self,
        task_info: Optional[dict] = None,
        # Phase 0 outputs
        signature: Optional[Any] = None,
        compute_pattern: Optional[str] = None,
        output_equals_input_shape: bool = False,
        shape_inference: Optional[Dict[str, str]] = None,
        functionality: Optional[str] = None,
        strategies: Optional[Dict[str, str]] = None,
        # Parallel branch outputs
        shape_inference_code: Optional[str] = None,  # PyTorch shape code for translation
        pybind_src: Optional[str] = None,
        kernel_src: Optional[str] = None,
        tiling_src: Optional[str] = None,
        operator_src: Optional[str] = None,
        # Joint planning
        joint_plan: Optional[dict] = None,
        joint_conversation: Optional[List[dict]] = None,
        # Knowledge retrieval results
        knowledge: Optional[dict] = None,
        # Debug state
        debug_history: Optional[List[dict]] = None,
        current_iteration: int = 0,
        is_done: bool = False,
        success: bool = False,
        # [Fix] 修正这里
        latest_analysis: Optional[dict] = None,
    ):
        # Basic info
        self.task_info = task_info or {}

        # Phase 0
        self.signature = signature
        self.compute_pattern = compute_pattern
        self.output_equals_input_shape = output_equals_input_shape
        self.shape_inference = shape_inference or {}
        self.functionality = functionality
        self.strategies = strategies or {}

        # Parallel branch outputs
        self.shape_inference_code = shape_inference_code
        self.pybind_src = pybind_src
        self.kernel_src = kernel_src
        self.tiling_src = tiling_src
        self.operator_src = operator_src

        # Joint planning
        self.joint_plan = joint_plan or {}
        self.joint_conversation = joint_conversation or []

        # Knowledge retrieval
        self.knowledge = knowledge or {}

        # Debug
        self.debug_history = debug_history or []
        self.current_iteration = current_iteration
        self.is_done = is_done
        self.success = success
        # [Fix] 修正这里
        self.latest_analysis = latest_analysis

    def to_pickle(self, file_path: str) -> None:
        """Save state to pickle file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file_path: str) -> "CANNIniterRunStateDict":
        """Load state from pickle file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)
