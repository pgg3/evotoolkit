# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from dataclasses import dataclass

import pytest

from evotoolkit.core import (
    EvaluationResult,
    Method,
    MethodState,
    PopulationMethodState,
    RunStore,
    Solution,
)
from evotoolkit.core.base_task import BaseTask
from evotoolkit.core.method_interface import BaseMethodInterface


class ConcreteTask(BaseTask):
    def _process_data(self, data):
        self.data = data
        self.task_info = {"name": "dummy"}

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        return EvaluationResult(valid=True, score=float(len(candidate_code)), additional_info={})

    def get_base_task_description(self) -> str:
        return "Dummy task."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("seed")


class DummyInterface(BaseMethodInterface):
    def parse_response(self, response_str: str) -> Solution:
        return Solution(response_str)


@dataclass
class DummyState(MethodState):
    iteration: int = 0


class DummyMethod(Method):
    algorithm_name = "dummy"

    def __init__(self, interface, output_path: str, *, max_iterations: int = 2, verbose: bool = False):
        self.max_iterations = max_iterations
        super().__init__(interface=interface, output_path=output_path, running_llm=None, verbose=verbose)

    def _create_state(self) -> DummyState:
        return DummyState(task_info=dict(self.task.task_info))

    def _bootstrap(self) -> None:
        seed = self._create_seed_solution()
        if seed is None:
            self.state.status = "failed"
            return
        self.state.sol_history.append(seed)
        self.state.status = "running"

    def _step(self) -> None:
        self.state.iteration += 1
        self.state.sol_history.append(
            Solution(
                f"step_{self.state.iteration}",
                evaluation_res=EvaluationResult(valid=True, score=float(self.state.iteration + 10), additional_info={}),
            )
        )

    def _should_stop(self) -> bool:
        return self.state.status == "failed" or self.state.iteration >= self.max_iterations

    def _select_best_solution(self) -> Solution | None:
        return self._get_best_sol(self.state.sol_history)


class TestMethodState:
    def test_method_state_defaults(self):
        state = MethodState()
        assert state.sol_history == []
        assert state.usage_history == {"sample": []}
        assert state.status == "created"
        assert state.bootstrapped is False

    def test_population_method_state_defaults(self):
        state = PopulationMethodState()
        assert state.generation == 0
        assert state.tot_sample_nums == 0
        assert state.population == []
        assert state.best_per_generation == []


class TestRunStore:
    def test_checkpoint_roundtrip_and_manifest(self, tmp_path):
        store = RunStore(str(tmp_path))
        state = DummyState(task_info={"name": "dummy"}, iteration=3, status="running", bootstrapped=True)

        store.save_checkpoint(
            state,
            algorithm="dummy",
            status="running",
            generation_or_iteration=3,
            sample_count=7,
            history_layout="iteration",
        )

        restored = store.load_checkpoint()
        manifest = store.load_manifest()

        assert isinstance(restored, DummyState)
        assert restored.iteration == 3
        assert manifest["algorithm"] == "dummy"
        assert manifest["sample_count"] == 7
        assert manifest["state_class"] == "DummyState"

    def test_generation_and_batch_history_roundtrip(self, tmp_path):
        store = RunStore(str(tmp_path))
        solution = Solution(
            "candidate",
            evaluation_res=EvaluationResult(valid=True, score=1.0, additional_info={"vector": [1, 2]}),
        )

        store.save_generation_history(0, [solution], [{"tokens": 1}], {"best_score": 1.0})
        store.save_batch_history(0, (0, 2), [solution], [{"tokens": 2}], {"valid_count": 1})
        store.save_usage_history({"sample": [{"tokens": 3}]})
        store.save_best_per_generation([{"generation": 0, "score": 1.0}])

        assert store.load_generation_history(0)["generation"] == 0
        assert store.load_batch_history(0)["batch_id"] == 0
        assert store.load_usage_history()["sample"][0]["tokens"] == 3
        assert store.load_best_per_generation()[0]["generation"] == 0


class TestMethodLifecycle:
    def test_run_bootstraps_and_returns_best_solution(self, tmp_path):
        task = ConcreteTask(data=None)
        interface = DummyInterface(task)
        method = DummyMethod(interface, str(tmp_path), max_iterations=2)

        best = method.run()

        assert best is not None
        assert best.evaluation_res.score == 12.0
        assert method.state.bootstrapped is True
        assert method.state.status == "completed"
        assert method.state.iteration == 2
        assert method.store.checkpoint_exists() is True

    def test_run_iteration_advances_single_step(self, tmp_path):
        task = ConcreteTask(data=None)
        interface = DummyInterface(task)
        method = DummyMethod(interface, str(tmp_path), max_iterations=3)

        method.run_iteration()

        assert method.state.bootstrapped is True
        assert method.state.iteration == 1
        assert method.best_solution is not None

    def test_load_checkpoint_restores_state(self, tmp_path):
        task = ConcreteTask(data=None)
        interface = DummyInterface(task)
        method = DummyMethod(interface, str(tmp_path), max_iterations=3)
        method.run_iteration()

        restored = DummyMethod(interface, str(tmp_path), max_iterations=3)
        restored.load_checkpoint()

        assert restored.state.iteration == 1
        assert restored.best_solution is not None

    def test_load_checkpoint_rejects_wrong_state_type(self, tmp_path):
        task = ConcreteTask(data=None)
        interface = DummyInterface(task)
        method = DummyMethod(interface, str(tmp_path), max_iterations=1)
        method.run_iteration()

        @dataclass
        class OtherState(MethodState):
            pass

        class OtherMethod(DummyMethod):
            def _create_state(self) -> MethodState:
                return OtherState(task_info=dict(self.task.task_info))

        restored = OtherMethod(interface, str(tmp_path), max_iterations=1)
        with pytest.raises(TypeError):
            restored.load_checkpoint()
