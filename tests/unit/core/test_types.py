import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from evotoolkit.core.types import EvaluationResult, Solution, TaskSpec, content_id


def _sol(evaluation=None):
    return Solution.create(
        files={"a.py": "x"},
        name="test",
        rationale="reason",
        parent_ids=["parent"],
        instruction="init",
        proposer="mock",
        call_id="call",
        usage={},
        created_step=0,
        evaluation=evaluation,
    )


def test_content_identity():
    assert content_id({"a.py": "x"}) == "5e29f12b6aef2eef"
    assert content_id({"a": "1", "b": "2"}) == content_id({"b": "2", "a": "1"})
    assert len({content_id({"a": "x"}), content_id({"b": "x"}), content_id({"a": "y"})}) == 3
    assert content_id({}) == "e3b0c44298fc1c14"


@pytest.mark.parametrize("evaluation", [None, EvaluationResult(True, 1.0, "ok")])
def test_solution_round_trip(evaluation):
    solution = _sol(evaluation)
    assert solution.id == content_id(solution.files)
    payload = solution.to_dict()
    payload["future_field"] = True
    if payload["evaluation"]:
        payload["evaluation"]["future_field"] = True
    assert Solution.from_dict(payload) == solution
    payload["id"] = "preserved-log-id"
    assert Solution.from_dict(payload).id == "preserved-log-id"


def test_task_defaults_unknown_fields_and_frozen():
    spec = TaskSpec.from_dict({"name": "test", "prompt": "do work", "modality": "python", "main_file": "a.py", "future": 1})
    assert spec.extras == {}
    assert TaskSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"


@pytest.mark.parametrize("evaluation, score", [(None, None), (EvaluationResult(False, 4.0, "bad"), None), (EvaluationResult(True, 0.0, "ok"), 0.0)])
def test_score(evaluation, score):
    assert _sol(evaluation).score == score


def test_numpy_values_are_plain_json():
    metrics = {"float": np.float64(1.2), "int": np.int64(2), "array": np.array([1, 2]), "bool": np.bool_(True)}
    payload = _sol(EvaluationResult(True, np.float64(3.0), "ok", metrics)).to_dict()
    assert json.loads(json.dumps(payload))["evaluation"]["metrics"] == {"float": 1.2, "int": 2, "array": [1, 2], "bool": True}
    assert EvaluationResult.from_dict({"valid": False, "score": None, "feedback": "bad", "future": 1}).cost_seconds == 0
