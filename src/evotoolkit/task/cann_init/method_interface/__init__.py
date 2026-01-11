# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANNInit task method interfaces"""

from .cann_initer import CANNIniterInterface
from .evoengineer import EvoEngineerCANNInterface
from .funsearch_interface import FunSearchCANNInterface
from .eoh_interface import EoHCANNInterface

__all__ = [
    "CANNIniterInterface",
    "EvoEngineerCANNInterface",
    "FunSearchCANNInterface",
    "EoHCANNInterface",
]
