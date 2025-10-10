# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Method interfaces for CUDA task optimization."""

from .eoh_interface import EoHCudaInterface
from .evoengineer_free_interface import EvoEngineerFreeCudaInterface
from .evoengineer_full_interface import EvoEngineerFullCudaInterface
from .evoengineer_insight_interface import EvoEngineerInsightCudaInterface
from .funsearch_interface import FunSearchCudaInterface

__all__ = [
    "EoHCudaInterface",
    "FunSearchCudaInterface",
    "EvoEngineerFullCudaInterface",
    "EvoEngineerInsightCudaInterface",
    "EvoEngineerFreeCudaInterface",
]
