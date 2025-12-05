# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANN Init task module for Ascend C operator generation.

This module generates Ascend C operators from Python reference implementations.
Distinct from future CANN optimization tasks.
"""

from .cann_init_task import CANNInitTask

__all__ = ["CANNInitTask"]
