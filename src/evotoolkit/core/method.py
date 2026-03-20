# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import os
from abc import ABC, abstractmethod
from typing import List

from .solution import Solution
from .state import MethodState
from .store import RunStore


class Method(ABC):
    """Thin template base class for method runtime lifecycle."""

    algorithm_name = "method"
    history_layout = "iteration"

    def __init__(
        self,
        interface,
        output_path: str = "./results",
        *,
        running_llm=None,
        verbose: bool = True,
    ):
        self.interface = interface
        self.task = interface.task
        self.running_llm = running_llm
        self.output_path = output_path
        self.verbose = verbose

        os.makedirs(self.output_path, exist_ok=True)
        self.store = RunStore(self.output_path)
        self.state = self._create_state()

    @property
    def best_solution(self) -> Solution | None:
        return self._select_best_solution()

    def run(self) -> Solution | None:
        self._ensure_initialized()

        while not self._should_stop():
            self.run_iteration()

        self._complete_if_needed()
        return self.best_solution

    def run_iteration(self) -> None:
        self._ensure_initialized()

        if self._should_stop():
            self._complete_if_needed()
            return

        self.state.status = "running"
        self._step()
        if self._should_stop() and self.state.status != "failed":
            self.state.status = "completed"
        self._persist_runtime()

    def save_checkpoint(self) -> None:
        self.store.save_checkpoint(
            self.state,
            algorithm=self.algorithm_name,
            status=self.state.status,
            generation_or_iteration=self._get_progress_index(),
            sample_count=self._get_sample_count(),
            history_layout=self.history_layout,
        )

    def load_checkpoint(self) -> None:
        loaded_state = self.store.load_checkpoint()
        if not isinstance(loaded_state, self.state.__class__):
            raise TypeError(
                f"Checkpoint state type mismatch: expected {self.state.__class__.__name__}, "
                f"got {loaded_state.__class__.__name__}"
            )
        self.state = loaded_state
        self.verbose_info(f"Loaded checkpoint from {self.store.state_file}")

    def checkpoint_exists(self) -> bool:
        return self.store.checkpoint_exists()

    def _ensure_initialized(self) -> None:
        if self.state.initialized:
            return

        self.state.status = "initializing"
        self._initialize()
        self.state.initialized = True
        if self.state.status == "initializing":
            self.state.status = "running"
        self._persist_runtime()

    def _complete_if_needed(self) -> None:
        if self.state.status in {"failed", "completed"}:
            return

        self.state.status = "completed"
        self._persist_runtime()

    def _persist_runtime(self) -> None:
        self._save_artifacts()
        self.save_checkpoint()

    def _get_progress_index(self) -> int:
        for attr in ("generation", "iteration", "current_batch_id", "tot_sample_nums"):
            value = getattr(self.state, attr, None)
            if isinstance(value, int):
                return value
        return 0

    def _get_sample_count(self) -> int:
        for attr in ("tot_sample_nums", "sample_count"):
            value = getattr(self.state, attr, None)
            if isinstance(value, int):
                return value
        return len(self.state.sol_history)

    def _save_artifacts(self) -> None:
        """Optional hook for algorithms to persist readable artifacts."""

    @staticmethod
    def _get_best_valid_sol(sol_list: List[Solution]) -> Solution:
        valid_sols = [
            sol
            for sol in sol_list
            if sol.evaluation_res is not None and sol.evaluation_res.valid and sol.evaluation_res.score is not None
        ]
        if not valid_sols:
            raise ValueError("No valid solutions available")
        return max(valid_sols, key=lambda x: x.evaluation_res.score)

    @staticmethod
    def _get_best_sol(sol_list: List[Solution]) -> Solution | None:
        if not sol_list:
            return None
        try:
            return Method._get_best_valid_sol(sol_list)
        except ValueError:
            return sol_list[0]

    def verbose_info(self, message: str) -> None:
        if self.verbose:
            print(message)

    def verbose_title(self, text: str, total_width: int = 60) -> None:
        if self.verbose:
            print("=" * total_width)
            print(text.center(total_width))
            print("=" * total_width)

    def verbose_stage(self, text: str, total_width: int = 60) -> None:
        if self.verbose:
            print("-" * total_width)
            print(text.center(total_width))
            print("-" * total_width)

    def verbose_gen(self, text: str, total_width: int = 60) -> None:
        if self.verbose:
            padding = (total_width - len(text)) // 2
            left_dashes = "-" * padding
            right_dashes = "-" * (total_width - len(text) - padding)
            print(left_dashes + text + right_dashes)

    @abstractmethod
    def _create_state(self) -> MethodState:
        raise NotImplementedError()

    @abstractmethod
    def _initialize(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _step(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _should_stop(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def _select_best_solution(self) -> Solution | None:
        raise NotImplementedError()
