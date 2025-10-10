# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import json
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .history_manager import HistoryManager


class BaseRunStateDict(ABC):
    def __init__(self, task_info: dict):
        self.task_info = task_info
        self._history_manager: Optional[HistoryManager] = None

    @staticmethod
    def _serialize_value(value):
        """Convert numpy arrays and other types to JSON-serializable format"""
        if isinstance(value, np.ndarray):
            return {
                "__numpy_array__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        elif isinstance(value, dict):
            return {k: BaseRunStateDict._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [BaseRunStateDict._serialize_value(item) for item in value]
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        else:
            # Basic types (str, int, float, bool, None) pass through
            # User-defined types will fail here - that's their responsibility
            return value

    @staticmethod
    def _deserialize_value(value):
        """Convert serialized numpy arrays back to original format"""
        if isinstance(value, dict):
            if value.get("__numpy_array__"):
                return np.array(value["data"], dtype=value["dtype"]).reshape(
                    value["shape"]
                )
            else:
                return {
                    k: BaseRunStateDict._deserialize_value(v) for k, v in value.items()
                }
        elif isinstance(value, list):
            return [BaseRunStateDict._deserialize_value(item) for item in value]
        else:
            return value

    @abstractmethod
    def to_json(self) -> dict:
        """Convert the run state to JSON-serializable dictionary"""
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict) -> "BaseRunStateDict":
        """Create instance from JSON data"""
        pass

    def to_json_file(self, file_path: str) -> None:
        """Save the run state to a JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, file_path: str) -> "BaseRunStateDict":
        """Load instance from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data)

    def init_history_manager(self, output_path: str) -> None:
        """初始化历史管理器"""
        self._history_manager = HistoryManager(output_path)

    @abstractmethod
    def save_current_history(self) -> None:
        """保存当前进度的历史记录（由子类实现具体逻辑）"""
        pass
