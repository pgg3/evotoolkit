# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from typing import List

from evotoolkit.core import Operator

from ..cuda_task import CudaTask
from .evoengineer_full_interface import EvoEngineerFullCudaInterface


class EvoEngineerInsightCudaInterface(EvoEngineerFullCudaInterface):
    def __init__(self, task_config: CudaTask):
        super().__init__(task_config)
        self.valid_require = 0

    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for CUDA optimization"""
        return [Operator("init", 0)]

    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for CUDA optimization"""
        return [Operator("init", 0)]
