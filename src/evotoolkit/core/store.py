# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import json
import os
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .solution import Solution


class RunStore:
    """Persistence layer for checkpoints and readable run artifacts."""

    format_version = "3.0.0"

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.checkpoint_dir = os.path.join(output_path, "checkpoint")
        self.history_dir = os.path.join(output_path, "history")
        self.summary_dir = os.path.join(output_path, "summary")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

    @property
    def state_file(self) -> str:
        return os.path.join(self.checkpoint_dir, "state.pkl")

    @property
    def manifest_file(self) -> str:
        return os.path.join(self.checkpoint_dir, "manifest.json")

    def checkpoint_exists(self) -> bool:
        return os.path.exists(self.state_file)

    def save_checkpoint(
        self,
        state: Any,
        *,
        algorithm: str,
        status: str,
        generation_or_iteration: int,
        sample_count: int,
        history_layout: str,
    ) -> dict:
        with open(self.state_file, "wb") as f:
            pickle.dump(state, f)

        manifest = {
            "format_version": self.format_version,
            "algorithm": algorithm,
            "state_class": state.__class__.__name__,
            "status": status,
            "generation_or_iteration": generation_or_iteration,
            "sample_count": sample_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "history_layout": history_layout,
            "state_file": os.path.relpath(self.state_file, self.output_path).replace("\\", "/"),
        }
        self._write_json(self.manifest_file, manifest)
        return manifest

    def load_checkpoint(self) -> Any:
        with open(self.state_file, "rb") as f:
            return pickle.load(f)

    def load_manifest(self) -> dict:
        if not os.path.exists(self.manifest_file):
            return {}
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "__numpy_array__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        if isinstance(value, dict):
            return {k: RunStore._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RunStore._serialize_value(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    @staticmethod
    def _solution_to_dict(solution: Solution) -> dict:
        payload = {
            "sol_string": solution.sol_string,
            "metadata": RunStore._serialize_value(solution.metadata.to_dict()),
            "evaluation_res": None,
        }
        if solution.evaluation_res is not None:
            payload["evaluation_res"] = {
                "valid": solution.evaluation_res.valid,
                "score": RunStore._serialize_value(solution.evaluation_res.score),
                "additional_info": RunStore._serialize_value(solution.evaluation_res.additional_info),
            }
        return payload

    def save_generation_history(
        self,
        generation: int,
        solutions: List[Solution],
        usage: List[Dict],
        statistics: Optional[Dict] = None,
    ) -> None:
        self._write_json(
            os.path.join(self.history_dir, f"gen_{generation}.json"),
            {
                "generation": generation,
                "solutions": [self._solution_to_dict(sol) for sol in solutions],
                "usage": self._serialize_value(usage),
                "statistics": self._serialize_value(statistics or {}),
            },
        )

    def load_generation_history(self, generation: int) -> Optional[Dict]:
        file_path = os.path.join(self.history_dir, f"gen_{generation}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_generations(self) -> List[int]:
        generations = []
        if not os.path.exists(self.history_dir):
            return generations
        for filename in os.listdir(self.history_dir):
            if filename.startswith("gen_") and filename.endswith(".json"):
                try:
                    generations.append(int(filename.removeprefix("gen_").removesuffix(".json")))
                except ValueError:
                    continue
        return sorted(generations)

    def save_batch_history(
        self,
        batch_id: int,
        sample_range: tuple,
        solutions: List[Solution],
        usage: List[Dict],
        metadata: Optional[Dict] = None,
    ) -> None:
        self._write_json(
            os.path.join(self.history_dir, f"batch_{batch_id:04d}.json"),
            {
                "batch_id": batch_id,
                "sample_range": list(sample_range),
                "solutions": [self._solution_to_dict(sol) for sol in solutions],
                "usage": self._serialize_value(usage),
                "metadata": self._serialize_value(metadata or {}),
            },
        )

    def load_batch_history(self, batch_id: int) -> Optional[Dict]:
        file_path = os.path.join(self.history_dir, f"batch_{batch_id:04d}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_batches(self) -> List[int]:
        batches = []
        if not os.path.exists(self.history_dir):
            return batches
        for filename in os.listdir(self.history_dir):
            if filename.startswith("batch_") and filename.endswith(".json"):
                try:
                    batches.append(int(filename.removeprefix("batch_").removesuffix(".json")))
                except ValueError:
                    continue
        return sorted(batches)

    def save_usage_history(self, usage_history: Dict) -> None:
        self._write_json(
            os.path.join(self.summary_dir, "usage_history.json"),
            self._serialize_value(usage_history),
        )

    def load_usage_history(self) -> Dict:
        file_path = os.path.join(self.summary_dir, "usage_history.json")
        if not os.path.exists(file_path):
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_best_per_generation(self, best_solutions: List[Dict]) -> None:
        self._write_json(
            os.path.join(self.summary_dir, "best_per_generation.json"),
            self._serialize_value(best_solutions),
        )

    def load_best_per_generation(self) -> List[Dict]:
        file_path = os.path.join(self.summary_dir, "best_per_generation.json")
        if not os.path.exists(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_summary(self, filename: str, payload: Any) -> None:
        self._write_json(os.path.join(self.summary_dir, filename), self._serialize_value(payload))

    @staticmethod
    def _write_json(file_path: str, payload: Any) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
