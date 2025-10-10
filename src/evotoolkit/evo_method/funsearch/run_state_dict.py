# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import json
import os
from typing import List

from evotoolkit.core import BaseRunStateDict, Solution


class FunSearchRunStateDict(BaseRunStateDict):
    def __init__(
        self,
        task_info: dict,
        tot_sample_nums: int = 0,
        sol_history: list[Solution] = None,
        database_file: str = None,
        is_done: bool = False,
        batch_size: int = 1,
    ):
        super().__init__(task_info)

        self.tot_sample_nums = tot_sample_nums
        self.sol_history = (
            sol_history or []
        )  # All solutions (valid/invalid, kept in memory)
        self.database_file = database_file  # Path to database JSON file
        self.is_done = is_done
        self.usage_history = {}
        self.batch_size = batch_size  # 每多少个sample保存一次历史

        # 当前批次新增的solution（用于历史保存）
        self.current_batch_solutions: List[Solution] = []
        self.current_batch_usage: List[dict] = []
        self.current_batch_start: int = 0

    def to_json(self) -> dict:
        """Convert the run state to JSON-serializable dictionary (only current state, no history)"""
        # 获取当前最优解
        current_best = None
        if self.sol_history:
            valid_sols = [
                s
                for s in self.sol_history
                if s.evaluation_res and s.evaluation_res.valid
            ]
            if valid_sols:
                best_sol = max(valid_sols, key=lambda x: x.evaluation_res.score)
                current_best = {
                    "score": best_sol.evaluation_res.score,
                    "sample_id": self.tot_sample_nums,
                    "sol_string": best_sol.sol_string,
                }

        return {
            "task_info": self._serialize_value(self.task_info),
            "database_file": self.database_file,
            "tot_sample_nums": self.tot_sample_nums,
            "batch_size": self.batch_size,
            "is_done": self.is_done,
            "current_best": current_best,
            "metadata": {
                "history_saved_in": "history/",
                "last_batch": (self.tot_sample_nums - 1) // self.batch_size
                if self.tot_sample_nums > 0
                else 0,
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> "FunSearchRunStateDict":
        """Create instance from JSON data (loads current state only, history loaded separately)"""
        instance = cls(
            task_info=cls._deserialize_value(data["task_info"]),
            tot_sample_nums=data.get("tot_sample_nums", 0),
            sol_history=[],  # History will be loaded separately if needed
            database_file=data.get("database_file"),
            is_done=data.get("is_done", False),
            batch_size=data.get("batch_size", 1),
        )
        return instance

    def save_current_history(self) -> None:
        """保存当前批次的历史记录"""
        if not self._history_manager:
            return

        if not self.current_batch_solutions:
            return

        # 检查是否应该保存：累积够batch_size个，或者算法结束
        should_save = (
            len(self.current_batch_solutions) >= self.batch_size
        ) or self.is_done

        if not should_save:
            return

        # 计算批次ID
        batch_id = (self.tot_sample_nums - 1) // self.batch_size

        # 计算样本范围
        sample_range = (self.current_batch_start, self.tot_sample_nums)

        # 元数据：包含island信息等
        metadata = {
            "valid_count": sum(
                1
                for s in self.current_batch_solutions
                if s.evaluation_res and s.evaluation_res.valid
            )
        }

        # 保存这个批次的历史
        self._history_manager.save_batch_history(
            batch_id=batch_id,
            sample_range=sample_range,
            solutions=self.current_batch_solutions,
            usage=self.current_batch_usage,
            metadata=metadata,
        )

        # 保存usage_history摘要
        self._history_manager.save_usage_history(self.usage_history)

        # 更新批次起始点并清空缓存
        self.current_batch_start = self.tot_sample_nums
        self.current_batch_solutions = []
        self.current_batch_usage = []

    def save_database_state(self, database_dict: dict, output_path: str) -> None:
        """Save database state to a separate JSON file"""
        if not self.database_file:
            # Generate database filename based on output path
            self.database_file = os.path.join(output_path, "programs_database.json")

        database_path = self.database_file
        if not os.path.isabs(database_path):
            database_path = os.path.join(output_path, database_path)

        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        with open(database_path, "w", encoding="utf-8") as f:
            json.dump(database_dict, f, indent=2, ensure_ascii=False)

    def load_database_state(self, output_path: str) -> dict:
        """Load database state from the separate JSON file"""
        if not self.database_file:
            return {}

        database_path = self.database_file
        if not os.path.isabs(database_path):
            database_path = os.path.join(output_path, database_path)

        if not os.path.exists(database_path):
            return {}

        try:
            with open(database_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load database state from {database_path}: {e}")
            return {}

    def has_database_state(self, output_path: str) -> bool:
        """Check if database state file exists"""
        if not self.database_file:
            return False

        database_path = self.database_file
        if not os.path.isabs(database_path):
            database_path = os.path.join(output_path, database_path)

        return os.path.exists(database_path)
