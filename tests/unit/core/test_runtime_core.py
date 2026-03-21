# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import math
from dataclasses import dataclass

import pytest

from evotoolkit.core import (
    EvaluationResult,
    IterationState,
    IterativeMethod,
    MethodInterface,
    MethodState,
    PopulationState,
    RunStore,
    Solution,
    Task,
    TaskSpec,
)


class ConcreteTask(Task):
    def build_spec(self, data) -> TaskSpec:
        return TaskSpec(name="dummy", prompt="Dummy task.", modality="generic", extras={"source": "test"})

    def evaluate(self, solution: Solution) -> EvaluationResult:
        return EvaluationResult(valid=True, score=float(len(solution.sol_string)), additional_info={})


class DummyInterface(MethodInterface):
    def parse_response(self, response_str: str) -> Solution:
        return Solution(response_str)


@dataclass
class DummyState(IterationState):
    pass


class DummyMethod(IterativeMethod):
    algorithm_name = "dummy"
    state_cls = DummyState

    def __init__(self, interface, output_path: str, *, max_iterations: int = 2, verbose: bool = False):
        self.max_iterations = max_iterations
        super().__init__(interface=interface, output_path=output_path, running_llm=None, verbose=verbose)

    def initialize_iteration(self) -> None:
        if self.state.sol_history:
            return

        initial_solution = Solution("initial")
        if initial_solution.evaluation_res is None:
            initial_solution.evaluation_res = self.task.evaluate(initial_solution)

        score = None if initial_solution.evaluation_res is None else initial_solution.evaluation_res.score
        if initial_solution.evaluation_res is None or not initial_solution.evaluation_res.valid or score is None or not math.isfinite(score):
            self.state.status = "failed"
            return
        self.state.sol_history.append(initial_solution)

    def step_iteration(self) -> None:
        self.state.iteration += 1
        self.state.sol_history.append(
            Solution(
                f"step_{self.state.iteration}",
                evaluation_res=EvaluationResult(valid=True, score=float(self.state.iteration + 10), additional_info={}),
            )
        )

    def should_stop_iteration(self) -> bool:
        return self.state.iteration >= self.max_iterations


class TestTaskRuntime:
    def test_task_builds_spec_on_init(self):
        task = ConcreteTask(data={"x": 1})
        assert task.data == {"x": 1}
        assert task.spec.name == "dummy"
        assert task.spec.prompt == "Dummy task."
        assert task.spec.modality == "generic"
        assert task.spec.extras["source"] == "test"

    def test_task_spec_copy_is_independent(self):
        spec = TaskSpec(name="a", prompt="b", modality="c", extras={"k": "v"})
        copied = spec.copy()
        copied.extras["k"] = "changed"
        assert spec.extras["k"] == "v"


class TestMethodState:
    def test_method_state_defaults(self):
        state = MethodState()
        assert state.task_spec == TaskSpec()
        assert state.sol_history == []
        assert state.usage_history == {"sample": []}
        assert state.status == "created"
        assert state.initialized is False

    def test_iteration_state_defaults(self):
        state = IterationState()
        assert state.iteration == 0
        assert state.get_progress_index() == 0

    def test_population_state_defaults(self):
        state = PopulationState()
        assert state.generation == 0
        assert state.sample_count == 0
        assert state.population == []
        assert state.best_per_generation == []


class TestRunStore:
    def test_checkpoint_roundtrip_and_manifest(self, tmp_path):
        store = RunStore(str(tmp_path))
        state = DummyState(
            task_spec=TaskSpec(name="dummy", prompt="Dummy task.", modality="generic"),
            iteration=3,
            status="running",
            initialized=True,
        )

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
        assert manifest["format_version"] == RunStore.format_version

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
    def test_run_initializes_and_returns_best_solution(self, tmp_path):
        task = ConcreteTask(data=None)
        interface = DummyInterface(task)
        method = DummyMethod(interface, str(tmp_path), max_iterations=2)

        best = method.run()

        assert best is not None
        assert best.evaluation_res.score == 12.0
        assert method.state.initialized is True
        assert method.state.status == "completed"
        assert method.state.iteration == 2
        assert method.store.checkpoint_exists() is True

    def test_run_iteration_advances_single_step(self, tmp_path):
        task = ConcreteTask(data=None)
        interface = DummyInterface(task)
        method = DummyMethod(interface, str(tmp_path), max_iterations=3)

        method.run_iteration()

        assert method.state.initialized is True
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
                return OtherState(task_spec=self.task.spec.copy())

        restored = OtherMethod(interface, str(tmp_path), max_iterations=1)
        with pytest.raises(TypeError):
            restored.load_checkpoint()
