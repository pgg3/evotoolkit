# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from evotoolkit.tools.llm import HttpsApi
from evotoolkit.core import Operator, EvoEngineerInterface, BaseConfig
from typing import Optional, Literal, List

class EvoEngineerConfig(BaseConfig):
    def __init__(
            self,
            interface: EvoEngineerInterface,
            output_path: str,
            running_llm: HttpsApi,
            verbose: bool = True,
            max_generations: int = 10,
            max_sample_nums: int = 45,
            pop_size: int = 5,
            num_samplers: int = 4,
            num_evaluators: int = 4,
            **kwargs
    ):
        super().__init__(interface, output_path, verbose)
        self.running_llm = running_llm


        self.max_generations = max_generations
        self.max_sample_nums = max_sample_nums
        self.pop_size = pop_size
        self.num_samplers = num_samplers
        self.num_evaluators = num_evaluators
        
        # Get operators from adapter
        self.init_operators = interface.get_init_operators()
        self.offspring_operators = interface.get_offspring_operators()
        
        # Validate required operators
        if not self.init_operators:
            raise ValueError("Adapter must provide at least one init operator")
        if not self.offspring_operators:
            raise ValueError("Adapter must provide at least one offspring operator")
        
        # Validate init operators have selection_size=0
        for op in self.init_operators:
            if op.selection_size != 0:
                raise ValueError(f"Init operator '{op.name}' must have selection_size=0, got {op.selection_size}")
    
    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators"""
        return self.init_operators
    
    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators"""
        return self.offspring_operators
    
    def get_all_operators(self) -> List[Operator]:
        """Get all operators"""
        return self.init_operators + self.offspring_operators