# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import os
from abc import ABC, abstractmethod
from typing import List, Type

from .base_config import BaseConfig
from .base_run_state_dict import BaseRunStateDict
from .solution import Solution


class Method(ABC):
    def __init__(self, config: BaseConfig):
        self.config = config
        self.run_state_dict = self._load_run_state_dict()
        # 初始化历史管理器
        self.run_state_dict.init_history_manager(self.config.output_path)
        self._save_run_state_dict()

    @abstractmethod
    def run(self, *args):
        raise NotImplementedError()

    def verbose_info(self, message: str):
        if self.config.verbose:
            print(message)

    def verbose_title(self, text: str, total_width: int = 60):
        """Display a centered title with equal signs above and below"""
        if self.config.verbose:
            print("=" * total_width)
            print(text.center(total_width))
            print("=" * total_width)

    def verbose_stage(self, text: str, total_width: int = 60):
        """Display a stage separator with dashes"""
        if self.config.verbose:
            print("-" * total_width)
            print(text.center(total_width))
            print("-" * total_width)

    def verbose_gen(self, text: str, total_width: int = 60):
        """Display text centered with dashes on both sides"""
        if self.config.verbose:
            padding = (total_width - len(text)) // 2
            left_dashes = "-" * padding
            right_dashes = "-" * (total_width - len(text) - padding)
            print(left_dashes + text + right_dashes)

    def _save_run_state_dict(self):
        """Save run state to file and history"""
        # 先保存历史
        self.run_state_dict.save_current_history()
        # 再保存当前状态 - use absolute path to avoid os.chdir() issues
        abs_output_path = os.path.abspath(self.config.output_path)
        self.run_state_dict.to_json_file(os.path.join(abs_output_path, "run_state.json"))

    def _load_run_state_dict(self) -> BaseRunStateDict | None:
        """Load run state from file"""
        # Use absolute path to avoid os.chdir() issues
        abs_output_path = os.path.abspath(self.config.output_path)
        run_state_file = os.path.join(abs_output_path, "run_state.json")

        run_state_class = self._get_run_state_class()
        if os.path.exists(run_state_file):
            self.verbose_info(f"Loading run state from file {run_state_file}")
            return run_state_class.from_json_file(run_state_file)
        else:
            run_state_dict = run_state_class(self.config.task.task_info)
            self.verbose_info("Initialized run state dict.")
            return run_state_dict

    @staticmethod
    def _get_best_valid_sol(sol_list: List[Solution]):
        valid_sols = []
        for sol in sol_list:
            if sol.evaluation_res is not None:
                if sol.evaluation_res.valid:
                    valid_sols.append(sol)

        # Return None if no valid solutions
        if not valid_sols:
            return None

        # Return the kernel with minimum runtime
        best_kernel = max(valid_sols, key=lambda x: x.evaluation_res.score)
        return best_kernel

    @staticmethod
    def _get_best_sol(sol_list: List[Solution]) -> Solution | None:
        """Get the best solution from a list. Returns None if list is empty."""
        if not sol_list:
            return None
        best_valid_sol = Method._get_best_valid_sol(sol_list)
        if best_valid_sol is not None:
            return best_valid_sol
        return sol_list[0]

    @abstractmethod
    def _get_run_state_class(self) -> Type[BaseRunStateDict]:
        """Return the algorithm-specific RunStateDict class"""
        pass
