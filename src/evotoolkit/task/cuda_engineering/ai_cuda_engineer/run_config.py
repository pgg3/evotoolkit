# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Configuration for AI CUDA Engineer - standalone implementation."""

from evotoolkit.tools.llm import HttpsApi
from ..evaluator import Evaluator
from typing import List


class AiCudaEngineerConfig:
    """Configuration class for AI CUDA Engineer (no inheritance)."""

    def __init__(
            self,
            task_info: dict,
            output_path: str,
            evaluator: Evaluator,
            conversion_llm: HttpsApi,
            translation_llm: HttpsApi,
            evo_llm_list: List[HttpsApi],
            embedding_llm: HttpsApi,
            rag_llm: HttpsApi,
            conversion_retry: int = 10,
            verbose: bool = True
    ):
        self.task_info = task_info
        self.output_path = output_path
        self.verbose = verbose
        self.evaluator = evaluator
        self.conversion_retry = conversion_retry
        self.conversion_llm = conversion_llm
        self.translation_llm = translation_llm
        self.evo_llm_list = evo_llm_list
        self.embedding_llm = embedding_llm
        self.rag_llm = rag_llm
