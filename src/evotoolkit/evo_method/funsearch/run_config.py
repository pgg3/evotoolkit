# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from evotoolkit.tools.llm import HttpsApi
from evotoolkit.core import BaseConfig, FunSearchInterface
from typing import Optional


class FunSearchConfig(BaseConfig):
    def __init__(
            self,
            interface: FunSearchInterface,
            output_path: str,
            running_llm: HttpsApi,
            verbose: bool = True,
            max_sample_nums: int = 45,
            num_islands: int = 5,
            max_population_size: int = 1000,
            num_samplers: int = 5,
            num_evaluators: int = 5,
            programs_per_prompt: int = 2,
            **kwargs  # Ignore extra arguments like max_generations, pop_size
    ):
        super().__init__(interface, output_path, verbose)
        self.running_llm = running_llm

        self.max_sample_nums = max_sample_nums
        self.num_islands = num_islands
        self.max_population_size = max_population_size
        self.num_samplers = num_samplers
        self.num_evaluators = num_evaluators
        self.programs_per_prompt = programs_per_prompt

        # Optionally log ignored parameters
        if kwargs and verbose:
            ignored = ', '.join(kwargs.keys())
            # Silently ignore, or uncomment to log:
            # print(f"FunSearchConfig: Ignoring parameters: {ignored}")