# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Shared fixtures and helpers for the evotoolkit test suite.

Provides:
- Concrete minimal task implementations (no external dependencies)
- Concrete minimal interface implementations
- MockLLM that returns preset code responses
- Common pytest fixtures
"""

from typing import List

import pytest

from evotoolkit.core import BaseConfig, EvaluationResult, Solution
from evotoolkit.core.base_task import BaseTask
from evotoolkit.core.method_interface import EoHInterface, EvoEngineerInterface, FunSearchInterface
from evotoolkit.core.operator import Operator
from evotoolkit.task.python_task.python_task import PythonTask

# ---------------------------------------------------------------------------
# Minimal concrete task implementations
# ---------------------------------------------------------------------------


class MinimalPythonTask(PythonTask):
    """Simplest possible PythonTask: evaluates whether code defines a function 'f'."""

    def _process_data(self, data):
        self.data = data
        self.task_info = {"name": "minimal", "description": "Test task"}

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        namespace = {}
        exec(candidate_code, namespace)  # noqa: S102
        if "f" not in namespace:
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "No function 'f'"})
        score = float(namespace["f"](1))
        return EvaluationResult(valid=True, score=score, additional_info={})

    def get_base_task_description(self) -> str:
        return "Implement a Python function f(x) that returns a numeric value."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("def f(x):\n    return x")


class AlwaysValidTask(BaseTask):
    """Task that always returns a valid result with a fixed score."""

    def _process_data(self, data):
        self.data = data
        self.task_info = {"name": "always_valid"}

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        return EvaluationResult(valid=True, score=1.0, additional_info={"code": candidate_code})

    def get_base_task_description(self) -> str:
        return "Always valid task."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("code_string")


class AlwaysInvalidTask(BaseTask):
    """Task that always returns an invalid result."""

    def _process_data(self, data):
        self.data = data
        self.task_info = {}

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "always invalid"})

    def get_base_task_description(self) -> str:
        return "Always invalid task."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("bad_code")


# ---------------------------------------------------------------------------
# Minimal concrete interface implementations
# ---------------------------------------------------------------------------

MOCK_PYTHON_CODE = "def f(x):\n    return x * 2"
MOCK_RESPONSE = "{Simple linear function}\n```python\ndef f(x):\n    return x * 2\n```"


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
        return Solution(MOCK_PYTHON_CODE, other_info={"algorithm": "test"})


class MinimalEvoEngineerInterface(EvoEngineerInterface):
    """Concrete EvoEngineer interface wrapping MinimalPythonTask."""

    def __init__(self, task: MinimalPythonTask):
        super().__init__(task)

    def get_init_operators(self) -> List[Operator]:
        return [Operator("init", selection_size=0)]

    def get_offspring_operators(self) -> List[Operator]:
        return [Operator("crossover", selection_size=2), Operator("mutate", selection_size=1)]

    def get_operator_prompt(self, operator_name, selected_individuals, current_best_sol, random_thoughts, **kwargs) -> List[dict]:
        return [{"role": "user", "content": f"Operator: {operator_name}"}]

    def parse_response(self, response_str: str) -> Solution:
        return Solution(MOCK_PYTHON_CODE, other_info={"name": "test", "thought": "test"})


class MinimalFunSearchInterface(FunSearchInterface):
    """Concrete FunSearch interface wrapping MinimalPythonTask."""

    def __init__(self, task: MinimalPythonTask):
        super().__init__(task)

    def make_init_sol(self) -> Solution:
        sol = self.task.make_init_sol_wo_other_info()
        return sol

    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        return [{"role": "user", "content": "Generate next solution."}]

    def parse_response(self, response_str: str) -> Solution:
        return Solution(MOCK_PYTHON_CODE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def valid_solution():
    return Solution(
        sol_string="def f(x): return x",
        other_info={"algorithm": "linear"},
        evaluation_res=EvaluationResult(valid=True, score=1.0, additional_info={}),
    )


@pytest.fixture
def invalid_solution():
    return Solution(
        sol_string="invalid code !!",
        other_info={},
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


@pytest.fixture
def base_config(eoh_interface, tmp_output):
    return BaseConfig(interface=eoh_interface, output_path=tmp_output, verbose=False)
