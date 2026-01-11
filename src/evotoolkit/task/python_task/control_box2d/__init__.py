# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Control Tasks with Box2D Physics for EvoToolkit.

This module provides tasks for evolving control policies in Gymnasium
Box2D environments including:
- LunarLander: Land a spacecraft on the moon
- (Future) BipedalWalker: Walk a bipedal robot
- (Future) CarRacing: Drive a car on a track
"""

from .lunar_lander_task import LunarLanderTask
from .method_interface import (
    EoHControlInterface,
    EvoEngineerControlInterface,
    FunSearchControlInterface,
)

__all__ = [
    "LunarLanderTask",
    "EoHControlInterface",
    "FunSearchControlInterface",
    "EvoEngineerControlInterface",
]
