# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Shared fixtures and helpers for the evotoolkit test suite.

Provides:
- Concrete minimal task implementations (no external dependencies)
- Concrete minimal interface implementations
- Common pytest fixtures
"""

from typing import List

import pytest

from evotoolkit.core import EvaluationResult, Solution, Task, TaskSpec
from evotoolkit.evo_method.eoh import EoHInterface
from evotoolkit.evo_method.evoengineer import EvoEngineerInterface, Operator
from evotoolkit.evo_method.funsearch import FunSearchInterface
from evotoolkit.task.python_task.python_task import PythonTask
from evotoolkit.task.string_optimization.string_task import StringTask


class MinimalPythonTask(PythonTask):
    """Simplest possible PythonTask: evaluates whether code defines a function 'f'."""

    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="minimal",
            prompt="Implement a Python function f(x) that returns a numeric value.",
            modality="python",
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        namespace = {}
        exec(candidate_code, namespace)  # noqa: S102
        if "f" not in namespace:
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "No function 'f'"})
        score = float(namespace["f"](1))
        return EvaluationResult(valid=True, score=score, additional_info={})


class AlwaysValidTask(Task):
    """Task that always returns a valid result with a fixed score."""

    def build_spec(self, data) -> TaskSpec:
        return TaskSpec(name="always_valid", prompt="Always valid task.", modality="generic")

    def evaluate(self, solution: Solution) -> EvaluationResult:
        return EvaluationResult(valid=True, score=1.0, additional_info={"code": solution.sol_string})


class AlwaysInvalidTask(Task):
    """Task that always returns an invalid result."""

    def build_spec(self, data) -> TaskSpec:
        return TaskSpec(name="always_invalid", prompt="Always invalid task.", modality="generic")

    def evaluate(self, solution: Solution) -> EvaluationResult:
        return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "always invalid"})


class MinimalStringTask(StringTask):
    """Minimal StringTask used to exercise generic string interfaces."""

    def build_string_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="minimal_string",
            prompt="Return a concise string solution.",
            modality="string",
        )

    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        score = float(len(candidate_string))
        return EvaluationResult(valid=True, score=score, additional_info={"length": len(candidate_string)})


MOCK_PYTHON_CODE = "def f(x):\n    return x * 2"


class MinimalEoHInterface(EoHInterface):
    """Concrete EoH interface wrapping MinimalPythonTask."""

    def __init__(self, task: MinimalPythonTask):
        super().__init__(task)

    def get_prompt_i1(self) -> List[dict]:
        return [{"role": "user", "content": "Generate a function f(x)."}]

    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        return [{"role": "user", "content": "Crossover."}]

    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        return [{"role": "user", "content": "Guided crossover."}]

    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        return [{"role": "user", "content": "Mutate."}]

    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        return [{"role": "user", "content": "Parameter mutate."}]

    def parse_response(self, response_str: str) -> Solution:
        return self.make_solution(MOCK_PYTHON_CODE, description="test")


class MinimalEvoEngineerInterface(EvoEngineerInterface):
    """Concrete EvoEngineer interface wrapping MinimalPythonTask."""

    def __init__(self, task: MinimalPythonTask):
        super().__init__(task)

    def get_init_operators(self) -> List[Operator]:
        return [Operator("init", selection_size=0)]

    def get_offspring_operators(self) -> List[Operator]:
        return [Operator("crossover", selection_size=2), Operator("mutate", selection_size=1)]

    def get_operator_prompt(
        self,
        operator_name,
        selected_individuals,
        current_best_sol,
        random_descriptions,
        **kwargs,
    ) -> List[dict]:
        return [{"role": "user", "content": f"Operator: {operator_name}"}]

    def parse_response(self, response_str: str) -> Solution:
        return self.make_solution(MOCK_PYTHON_CODE, name="test", description="test")


class MinimalFunSearchInterface(FunSearchInterface):
    """Concrete FunSearch interface wrapping MinimalPythonTask."""

    def __init__(self, task: MinimalPythonTask):
        super().__init__(task)

    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        return [{"role": "user", "content": "Generate next solution."}]

    def parse_response(self, response_str: str) -> Solution:
        return self.make_solution(MOCK_PYTHON_CODE)


@pytest.fixture
def minimal_task():
    return MinimalPythonTask(data=None)


@pytest.fixture
def always_valid_task():
    return AlwaysValidTask(data=None)


@pytest.fixture
def always_invalid_task():
    return AlwaysInvalidTask(data=None)


@pytest.fixture
def minimal_string_task():
    return MinimalStringTask(data=None)


@pytest.fixture
def valid_solution():
    return Solution(
        sol_string="def f(x): return x",
        metadata={"description": "linear"},
        evaluation_res=EvaluationResult(valid=True, score=1.0, additional_info={}),
    )


@pytest.fixture
def invalid_solution():
    return Solution(
        sol_string="invalid code !!",
        metadata={},
        evaluation_res=EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "invalid"}),
    )


@pytest.fixture
def solution_list(valid_solution, invalid_solution):
    sol_medium = Solution(
        sol_string="def f(x): return x + 1",
        evaluation_res=EvaluationResult(valid=True, score=2.0, additional_info={}),
    )
    sol_best = Solution(
        sol_string="def f(x): return x * 3",
        evaluation_res=EvaluationResult(valid=True, score=3.0, additional_info={}),
    )
    return [valid_solution, sol_medium, invalid_solution, sol_best]


@pytest.fixture
def eoh_interface(minimal_task):
    return MinimalEoHInterface(minimal_task)


@pytest.fixture
def evoengineer_interface(minimal_task):
    return MinimalEvoEngineerInterface(minimal_task)


@pytest.fixture
def funsearch_interface(minimal_task):
    return MinimalFunSearchInterface(minimal_task)


@pytest.fixture
def tmp_output(tmp_path):
    return str(tmp_path / "output")
