# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANNIniter Interface: 专员 Prompt 管理"""

from .prompts import (
    Phase0PromptMixin,
    PybindPromptMixin,
    JointPromptMixin,
    DebugPromptMixin,
)


class CANNIniterInterface(
    Phase0PromptMixin,
    PybindPromptMixin,
    JointPromptMixin,
    DebugPromptMixin,
):
    """
    CANNIniter 专员 Prompt 管理

    组合所有专员的 Prompt 接口：
    - Phase0PromptMixin: 计算模式识别
    - PybindPromptMixin: Pybind 专员
    - JointPromptMixin: Kernel + Tiling 联合分支
    - DebugPromptMixin: Debug 专员

    注意: knowledge_base 由 CANNIniter 持有，Interface 专注 prompt engineering
    """
    pass
