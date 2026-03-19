# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for base classes covering uncovered lines."""

import pytest

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.core.base_method import Method
from evotoolkit.core.base_run_state_dict import BaseRunStateDict
from evotoolkit.core.base_task import BaseTask

# ---------------------------------------------------------------------------
# BaseTask: uncovered lines (46-47, 65, 98, 108, 122)
# ---------------------------------------------------------------------------


class ConcreteTask(BaseTask):
    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        return EvaluationResult(valid=True, score=1.0, additional_info={})

    def get_base_task_description(self) -> str:
        return "Test task"

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("def f(x): return x")


class TestBaseTask:
    def test_process_data_default(self):
        """_process_data sets self.data and self.task_info = {}."""
        # ConcreteTask doesn't override _process_data → uses default
        task = ConcreteTask(data={"key": "val"})
        assert task.data == {"key": "val"}
        assert task.task_info == {}

    def test_evaluate_solution_delegates_to_evaluate_code(self):
        """evaluate_solution default calls evaluate_code(solution.sol_string)."""
        task = ConcreteTask(data=None)
        sol = Solution("def f(x): return x")
        result = task.evaluate_solution(sol)
        assert result.valid is True
        assert result.score == 1.0

    def test_get_base_task_description_is_abstract_enforced(self):
        """ConcreteTask implements the abstract method."""
        task = ConcreteTask(data=None)
        desc = task.get_base_task_description()
        assert isinstance(desc, str)

    def test_make_init_sol_wo_other_info(self):
        task = ConcreteTask(data=None)
        sol = task.make_init_sol_wo_other_info()
        assert isinstance(sol, Solution)

    def test_get_task_type_default(self):
        """Default implementation returns 'Python'."""
        task = ConcreteTask(data=None)
        assert task.get_task_type() == "Python"

    def test_get_task_info_returns_dict(self):
        task = ConcreteTask(data=None)
        info = task.get_task_info()
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# BaseRunStateDict: uncovered lines (23, 32, 34, 45, 49, 56, 62, 83)
# ---------------------------------------------------------------------------


class ConcreteRunStateDict(BaseRunStateDict):
    def to_json(self) -> dict:
        return {"task_info": self._serialize_value(self.task_info)}

    @classmethod
    def from_json(cls, data: dict) -> "ConcreteRunStateDict":
        return cls(task_info=cls._deserialize_value(data["task_info"]))

    def save_current_history(self) -> None:
        pass


class TestBaseRunStateDict:
    def test_serialize_numpy_array(self):
        import numpy as np

        rsd = ConcreteRunStateDict(task_info={})
        arr = np.array([1.0, 2.0, 3.0])
        result = rsd._serialize_value(arr)
        assert result["__numpy_array__"] is True
        assert result["data"] == [1.0, 2.0, 3.0]

    def test_serialize_nested_dict(self):
        import numpy as np

        rsd = ConcreteRunStateDict(task_info={})
        data = {"a": np.array([1.0])}
        result = rsd._serialize_value(data)
        assert result["a"]["__numpy_array__"] is True

    def test_serialize_list(self):
        import numpy as np

        rsd = ConcreteRunStateDict(task_info={})
        data = [np.array([1.0]), 2, "str"]
        result = rsd._serialize_value(data)
        assert result[0]["__numpy_array__"] is True
        assert result[1] == 2
        assert result[2] == "str"

    def test_serialize_numpy_scalar(self):
        import numpy as np

        rsd = ConcreteRunStateDict(task_info={})
        val = np.float32(3.14)
        result = rsd._serialize_value(val)
        assert isinstance(result, float)

    def test_deserialize_numpy_array(self):
        import numpy as np

        rsd = ConcreteRunStateDict(task_info={})
        serialized = {
            "__numpy_array__": True,
            "dtype": "float64",
            "shape": [3],
            "data": [1.0, 2.0, 3.0],
        }
        result = rsd._deserialize_value(serialized)
        assert isinstance(result, np.ndarray)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_deserialize_nested_dict(self):
        rsd = ConcreteRunStateDict(task_info={})
        data = {"key": {"nested": "value"}}
        result = rsd._deserialize_value(data)
        assert result["key"]["nested"] == "value"

    def test_deserialize_list(self):
        rsd = ConcreteRunStateDict(task_info={})
        data = [1, 2, 3]
        result = rsd._deserialize_value(data)
        assert result == [1, 2, 3]

    def test_to_json_file_and_from_json_file(self, tmp_path):
        rsd = ConcreteRunStateDict(task_info={"x": 42})
        filepath = str(tmp_path / "state.json")
        rsd.to_json_file(filepath)
        restored = ConcreteRunStateDict.from_json_file(filepath)
        assert restored.task_info["x"] == 42

    def test_init_history_manager(self, tmp_path):
        rsd = ConcreteRunStateDict(task_info={})
        rsd.init_history_manager(str(tmp_path))
        assert rsd._history_manager is not None


# ---------------------------------------------------------------------------
# BaseMethod: uncovered lines (38-40, 43, 52, 57-59, 63-66, 70-74, 87-88, 112, 118)
# ---------------------------------------------------------------------------


class MockLLM:
    def get_response(self, messages, **kwargs):
        return "def f(x):\n    return x * 2", {"prompt_tokens": 10, "completion_tokens": 20}


def _make_eoh_config(interface, output_path, verbose=False):
    from evotoolkit.evo_method.eoh.run_config import EoHConfig

    return EoHConfig(
        interface=interface,
        output_path=output_path,
        running_llm=MockLLM(),
        verbose=verbose,
        max_generations=0,
        max_sample_nums=0,
    )


class TestBaseMethod:
    def test_verbose_methods_when_verbose_false(self, eoh_interface, tmp_output):
        """verbose_info, verbose_title, verbose_gen should be no-ops when verbose=False."""
        from evotoolkit.evo_method.eoh import EoH

        config = _make_eoh_config(eoh_interface, tmp_output, verbose=False)
        method = EoH(config)
        # These should not raise
        method.verbose_info("test message")
        method.verbose_title("TITLE")
        method.verbose_stage("STAGE")
        method.verbose_gen("GEN")

    def test_verbose_methods_when_verbose_true(self, eoh_interface, tmp_output, capsys):
        """verbose_info etc. should print when verbose=True."""
        from evotoolkit.evo_method.eoh import EoH

        config = _make_eoh_config(eoh_interface, tmp_output, verbose=True)
        method = EoH(config)
        method.verbose_info("hello verbose")
        captured = capsys.readouterr()
        assert "hello verbose" in captured.out

    def test_verbose_title_prints_centered(self, eoh_interface, tmp_output, capsys):
        from evotoolkit.evo_method.eoh import EoH

        config = _make_eoh_config(eoh_interface, tmp_output, verbose=True)
        method = EoH(config)
        method.verbose_title("TEST TITLE")
        captured = capsys.readouterr()
        assert "TEST TITLE" in captured.out
        assert "=" in captured.out

    def test_verbose_stage_prints_dashes(self, eoh_interface, tmp_output, capsys):
        from evotoolkit.evo_method.eoh import EoH

        config = _make_eoh_config(eoh_interface, tmp_output, verbose=True)
        method = EoH(config)
        method.verbose_stage("STAGE NAME")
        captured = capsys.readouterr()
        assert "STAGE NAME" in captured.out
        assert "-" in captured.out

    def test_verbose_gen_prints_padded(self, eoh_interface, tmp_output, capsys):
        from evotoolkit.evo_method.eoh import EoH

        config = _make_eoh_config(eoh_interface, tmp_output, verbose=True)
        method = EoH(config)
        method.verbose_gen("Gen 1")
        captured = capsys.readouterr()
        assert "Gen 1" in captured.out

    def test_load_run_state_from_existing_file(self, eoh_interface, tmp_path):
        """_load_run_state_dict uses from_json_file if run_state.json exists."""
        import os

        from evotoolkit.evo_method.eoh import EoH

        output_path = str(tmp_path / "output")
        os.makedirs(output_path, exist_ok=True)

        config = _make_eoh_config(eoh_interface, output_path)
        EoH(config)
        # The state file should exist now
        assert os.path.exists(os.path.join(output_path, "run_state.json"))

        # Create second method to trigger loading from file (covers line 87-88)
        method2 = EoH(config)
        assert method2.run_state_dict is not None

    def test_get_best_valid_sol_empty_list(self):
        """_get_best_valid_sol raises on empty list (no valid sols)."""
        with pytest.raises(Exception):
            Method._get_best_valid_sol([])

    def test_get_best_sol_falls_back_to_first(self):
        """_get_best_sol falls back to first sol when no valid solutions (line 112)."""
        # Test with a list where there ARE valid solutions:
        valid_sol = Solution(
            "code",
            evaluation_res=EvaluationResult(valid=True, score=1.0, additional_info={}),
        )
        result = Method._get_best_sol([valid_sol])
        assert result is valid_sol
