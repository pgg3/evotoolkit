# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import os
from abc import ABC, abstractmethod
from typing import List, Type

import numpy as np

from evotoolkit.core import BaseConfig, Solution

from .base_run_state_dict import BaseRunStateDict


class Method(ABC):
    def __init__(self, config: BaseConfig):
        self.config = config
        self.run_state_dict = self._load_run_state_dict()
        # 初始化历史管理器
        self.run_state_dict.init_history_manager(self.config.output_path)
        self._save_run_state_dict()

    def _get_init_sol(self):
        # Try to create and evaluate initial solution up to 3 times
        initial_sol = None
        for attempt in range(3):
            try:
                candidate_sol = self.config.interface.make_init_sol()
                if candidate_sol.evaluation_res is None:
                    candidate_sol.evaluation_res = self.config.task.evaluate_code(
                        candidate_sol.sol_string
                    )

                if (
                    candidate_sol.evaluation_res is not None
                    and candidate_sol.evaluation_res.valid
                ):
                    if (not np.isinf(candidate_sol.evaluation_res.score)) and (
                        candidate_sol.evaluation_res.score > -np.inf
                    ):
                        initial_sol = candidate_sol
                        break
                else:
                    self.verbose_info(
                        f"Initial solution attempt {attempt + 1} failed: invalid evaluation result"
                    )
            except Exception as e:
                self.verbose_info(
                    f"Initial solution attempt {attempt + 1} failed with exception: {e}"
                )

        if initial_sol is None:
            print(
                "Warning: Failed to create valid initial solution after 3 attempts. Exiting."
            )
        return initial_sol

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
        # 再保存当前状态
        self.run_state_dict.to_json_file(
            os.path.join(self.config.output_path, "run_state.json")
        )

    def _load_run_state_dict(self) -> BaseRunStateDict | None:
        """Load run state from file"""
        run_state_class = self._get_run_state_class()
        if os.path.exists(os.path.join(self.config.output_path, "run_state.json")):
            self.verbose_info(
                f"Loading run state from file {os.path.join(self.config.output_path, 'run_state.json')}"
            )
            return run_state_class.from_json_file(
                os.path.join(self.config.output_path, "run_state.json")
            )
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

        # Return the kernel with minimum runtime
        best_kernel = max(valid_sols, key=lambda x: x.evaluation_res.score)
        return best_kernel

    @staticmethod
    def _get_best_sol(sol_list: List[Solution]):
        best_valid_sol = Method._get_best_valid_sol(sol_list)
        if best_valid_sol is not None:
            best_sol = best_valid_sol
        else:
            best_sol = sol_list[0]
        return best_sol

    @abstractmethod
    def _get_run_state_class(self) -> Type[BaseRunStateDict]:
        """Return the algorithm-specific RunStateDict class"""
        pass
