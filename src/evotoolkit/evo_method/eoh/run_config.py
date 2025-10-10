# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from evotoolkit.tools.llm import HttpsApi
from evotoolkit.core import BaseConfig, EoHInterface

class EoHConfig(BaseConfig):
    def __init__(
            self,
            interface: EoHInterface,
            output_path: str,
            running_llm: HttpsApi,
            verbose: bool = True,
            max_generations: int = 10,
            max_sample_nums: int = 45,
            pop_size: int = 5,
            selection_num: int = 2,
            use_e2_operator: bool = True,
            use_m1_operator: bool = True,
            use_m2_operator: bool = True,
            num_samplers: int = 5,
            num_evaluators: int = 5,
    ):
        super().__init__(interface, output_path, verbose)
        self.running_llm = running_llm
        self.max_generations = max_generations
        self.max_sample_nums = max_sample_nums
        self.pop_size = pop_size
        self.selection_num = selection_num
        self.use_e2_operator = use_e2_operator
        self.use_m1_operator = use_m1_operator
        self.use_m2_operator = use_m2_operator
        self.num_samplers = num_samplers
        self.num_evaluators = num_evaluators