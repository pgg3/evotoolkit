# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Run state dictionary for AI CUDA Engineer - standalone implementation."""

import json
import numpy as np
from typing import Literal, Optional, List
from evotoolkit.core import HistoryManager


class AiCudaEngineerRunStateDict:
    """Run state for AI CUDA Engineer (no inheritance)."""

    def __init__(
            self,
            task_info: dict,
            run_stage: Literal["0", "1", "2", "3"] = "0",
            evo_gen_i: int = 0,
            optimization_history: list = None,
            usage_history: dict = None,
            is_done: bool = False
    ):
        self.task_info = task_info
        self.run_stage = run_stage  # 0: conversion, 1:translation, 2:evolution, 3:rag
        self.evo_gen_i = evo_gen_i
        self.optimization_history = optimization_history or []
        self.usage_history = usage_history or {}
        self.is_done = is_done
        self._history_manager: Optional[HistoryManager] = None

        # Track current stage/generation data for history saving
        self.current_stage_optimizations: List[dict] = []
        self.current_stage_usage: List[dict] = []

    def init_history_manager(self, output_path: str) -> None:
        """Initialize history manager."""
        self._history_manager = HistoryManager(output_path)

    def add_optimization_result(self, entry: dict) -> None:
        """Add optimization result to both history and current stage cache."""
        self.optimization_history.append(entry)
        self.current_stage_optimizations.append(entry)

    def add_usage_result(self, stage: str, usage: dict) -> None:
        """Add usage result to both history and current stage cache."""
        if stage not in self.usage_history:
            self.usage_history[stage] = []
        self.usage_history[stage].append(usage)
        self.current_stage_usage.append(usage)

    def get_best_kernel(self) -> dict:
        """Get the best performing valid kernel from optimization history."""
        valid_kernels = [k for k in self.optimization_history
                        if k.get("runtime") is not None and k["runtime"] != float('inf')]
        if not valid_kernels:
            return None
        return min(valid_kernels, key=lambda x: x["runtime"])

    def save_current_history(self) -> None:
        """Save current stage/generation history to separate files."""
        if not self._history_manager:
            return

        if not self.current_stage_optimizations:
            return

        import os
        import json

        # Calculate statistics
        valid_kernels = [k for k in self.current_stage_optimizations
                        if k.get("runtime") is not None and k["runtime"] != float('inf')]
        statistics = {
            'total_kernels': len(self.current_stage_optimizations),
            'valid_kernels': len(valid_kernels),
            'valid_rate': len(valid_kernels) / len(self.current_stage_optimizations) if self.current_stage_optimizations else 0,
        }

        if valid_kernels:
            runtimes = [k['runtime'] for k in valid_kernels]
            statistics['avg_runtime'] = sum(runtimes) / len(runtimes)
            statistics['best_runtime'] = min(runtimes)
            statistics['worst_runtime'] = max(runtimes)

        # Determine generation ID based on stage
        if self.run_stage == "0":
            gen_id = "stage0_conversion"
        elif self.run_stage == "1":
            gen_id = "stage1_translation"
        elif self.run_stage == "2":
            gen_id = f"stage2_evo_gen{self.evo_gen_i - 1}"  # Save previous generation
        elif self.run_stage == "3":
            gen_id = "stage3_rag"
        else:
            gen_id = f"stage{self.run_stage}"

        # Prepare generation data
        generation_data = {
            'generation': gen_id,
            'optimizations': self.current_stage_optimizations,
            'usage': self.current_stage_usage,
            'statistics': statistics
        }

        # Save to history directory (directly, not using HistoryManager's method)
        history_dir = os.path.join(self._history_manager.output_path, 'history')
        os.makedirs(history_dir, exist_ok=True)

        history_file = os.path.join(history_dir, f'gen_{gen_id}.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(generation_data, f, indent=2, ensure_ascii=False)

        # Save usage_history summary using HistoryManager
        self._history_manager.save_usage_history(self.usage_history)

        # Clear current stage cache
        self.current_stage_optimizations = []
        self.current_stage_usage = []

    @staticmethod
    def _serialize_value(value):
        """Convert numpy arrays and other types to JSON-serializable format."""
        if isinstance(value, np.ndarray):
            return {
                "__numpy_array__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist()
            }
        elif isinstance(value, dict):
            return {k: AiCudaEngineerRunStateDict._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [AiCudaEngineerRunStateDict._serialize_value(item) for item in value]
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        else:
            return value

    @staticmethod
    def _deserialize_value(value):
        """Convert serialized numpy arrays back to original format."""
        if isinstance(value, dict):
            if value.get("__numpy_array__"):
                return np.array(value["data"], dtype=value["dtype"]).reshape(value["shape"])
            else:
                return {k: AiCudaEngineerRunStateDict._deserialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [AiCudaEngineerRunStateDict._deserialize_value(item) for item in value]
        else:
            return value

    def to_json(self) -> dict:
        """Convert the run state to JSON-serializable dictionary (only current state, no full history)."""
        # Get current best kernel and serialize it completely
        best_kernel = self.get_best_kernel()
        current_best = self._serialize_value(best_kernel) if best_kernel else None

        return {
            'task_info': self.task_info,
            'usage_history': self.usage_history,
            'run_stage': self.run_stage,
            'evo_gen_i': self.evo_gen_i,
            'current_best': current_best,
            'is_done': self.is_done,
            'metadata': {
                'history_saved_in': 'history/',
                'total_optimization_entries': len(self.optimization_history)
            }
        }

    @classmethod
    def from_json(cls, data: dict) -> 'AiCudaEngineerRunStateDict':
        """Create instance from JSON data (loads current state only)."""
        # Support multiple formats:
        # - New format: current_best only
        # - Old format: top_10_kernels or optimization_history
        current_best = data.get('current_best')
        if current_best:
            optimization_history = [current_best]
        else:
            optimization_history = data.get('top_10_kernels', data.get('optimization_history', []))

        instance = cls(
            task_info=data['task_info'],
            run_stage=data.get('run_stage', "0"),  # type: ignore
            evo_gen_i=data.get('evo_gen_i', 0),
            optimization_history=optimization_history,
            usage_history=data.get('usage_history', {}),
            is_done=data.get('is_done', False)
        )
        return instance

    def to_json_file(self, file_path: str) -> None:
        """Save the run state to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'AiCudaEngineerRunStateDict':
        """Load instance from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_json(data)
