# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import math
import os
from abc import ABC, abstractmethod
from typing import List

from .solution import Solution
from .state import IterationState, MethodState, PopulationState
from .store import RunStore


class Method(ABC):
    """Thin template base class for method runtime lifecycle."""

    algorithm_name = "method"
    history_layout = "iteration"
    state_cls = MethodState

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
            raise TypeError(f"Checkpoint state type mismatch: expected {self.state.__class__.__name__}, got {loaded_state.__class__.__name__}")
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
        return self.state.get_progress_index()

    def _get_sample_count(self) -> int:
        return self.state.get_sample_count()

    def _save_artifacts(self) -> None:
        """Optional hook for algorithms to persist readable artifacts."""

    @staticmethod
    def _get_best_valid_sol(sol_list: List[Solution]) -> Solution:
        valid_sols = [sol for sol in sol_list if sol.evaluation_res is not None and sol.evaluation_res.valid and sol.evaluation_res.score is not None]
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

    def _create_state(self) -> MethodState:
        return self.state_cls(task_spec=self.task.spec.copy())

    @abstractmethod
    def _initialize(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _step(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _should_stop(self) -> bool:
        raise NotImplementedError()

    def _select_best_solution(self) -> Solution | None:
        return self._get_best_sol(self.state.sol_history)


class IterativeMethod(Method):
    """Convenience base class for custom step-wise search methods."""

    state_cls = IterationState
    startup_title: str | None = None

    def _initialize(self) -> None:
        if self.startup_title:
            self.verbose_title(self.startup_title)
        self.prepare_initialization()
        self.initialize_iteration()

    def _step(self) -> None:
        self.step_iteration()

    def _should_stop(self) -> bool:
        return self.state.status == "failed" or self.should_stop_iteration()

    def prepare_initialization(self) -> None:
        """Optional hook that runs before method-specific initialization."""

    def initialize_iteration(self) -> None:
        """Optional hook for method-specific initialization."""

    @abstractmethod
    def step_iteration(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def should_stop_iteration(self) -> bool:
        raise NotImplementedError()


class PopulationMethod(IterativeMethod):
    """Convenience base class for generation-based population methods."""

    history_layout = "generation"
    state_cls = PopulationState

    def _register_population_solution(self, solution: Solution) -> None:
        self.state.sol_history.append(solution)
        self.state.population.append(solution)
        self.state.current_generation_solutions.append(solution)
        self.state.sample_count += 1

    def _record_generation_usage(self, usage: dict) -> None:
        self.state.usage_history["sample"].append(usage)
        self.state.current_generation_usage.append(usage)

    def _get_valid_population(self, population: List[Solution] | None = None) -> List[Solution]:
        active_population = self.state.population if population is None else population
        return [sol for sol in active_population if sol.evaluation_res and sol.evaluation_res.valid]

    def _select_ranked_individuals(self, num_select: int) -> List[Solution]:
        import numpy as np

        if num_select <= 0:
            return []

        funcs = [
            sol
            for sol in self.state.population
            if sol.evaluation_res
            and sol.evaluation_res.valid
            and sol.evaluation_res.score is not None
            and not math.isinf(sol.evaluation_res.score)
            and not math.isnan(sol.evaluation_res.score)
        ]

        if not funcs:
            return self.state.population[:num_select] if self.state.population else []

        ranked = sorted(funcs, key=lambda f: f.evaluation_res.score, reverse=True)
        probabilities = np.array([1 / (r + len(ranked)) for r in range(len(ranked))])
        probabilities = probabilities / np.sum(probabilities)

        selected = []
        for _ in range(min(num_select, len(ranked))):
            selected.append(np.random.choice(ranked, p=probabilities))
        return selected

    def _trim_population(self, max_population_size: int) -> None:
        if len(self.state.population) <= max_population_size:
            return

        valid_solutions = self._get_valid_population()
        invalid_solutions = [sol for sol in self.state.population if sol not in valid_solutions]

        valid_solutions.sort(
            key=lambda x: x.evaluation_res.score if x.evaluation_res and x.evaluation_res.score is not None else float("-inf"),
            reverse=True,
        )

        new_population = valid_solutions[: min(len(valid_solutions), max_population_size)]
        remaining_slots = max_population_size - len(new_population)
        if remaining_slots > 0 and invalid_solutions:
            new_population.extend(invalid_solutions[-remaining_slots:])

        self.state.population = new_population
        valid_count = len(self._get_valid_population(new_population))
        self.verbose_info(f"Population managed: {len(new_population)} total ({valid_count} valid, {len(new_population) - valid_count} invalid)")

    def _save_artifacts(self) -> None:
        if not self.state.current_generation_solutions:
            return

        valid_sols = [s for s in self.state.current_generation_solutions if s.evaluation_res and s.evaluation_res.valid]
        statistics = {
            "total_solutions": len(self.state.current_generation_solutions),
            "valid_solutions": len(valid_sols),
            "valid_rate": len(valid_sols) / len(self.state.current_generation_solutions),
        }
        if valid_sols:
            scores = [s.evaluation_res.score for s in valid_sols]
            statistics["avg_score"] = sum(scores) / len(scores)
            statistics["best_score"] = max(scores)
            statistics["worst_score"] = min(scores)

        completed_generation = max(self.state.generation - 1, 0)
        self.store.save_generation_history(
            generation=completed_generation,
            solutions=self.state.current_generation_solutions,
            usage=self.state.current_generation_usage,
            statistics=statistics,
        )
        self.store.save_usage_history(self.state.usage_history)

        best_solution = self._select_best_solution()
        if best_solution and best_solution.evaluation_res:
            self.state.best_per_generation.append(
                {
                    "generation": completed_generation,
                    "score": best_solution.evaluation_res.score,
                    "sol_string": best_solution.sol_string,
                }
            )
            self.store.save_best_per_generation(self.state.best_per_generation)

        self.state.current_generation_solutions = []
        self.state.current_generation_usage = []
