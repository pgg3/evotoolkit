# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Additional branch coverage for application-facing tasks."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from evotoolkit.task.python_task.adversarial_attack import AdversarialAttackTask
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask


def _datasets_with_n_inputs(n_inputs: int):
    if n_inputs == 4:
        train_inputs = np.arange(24, dtype=float).reshape(6, 4)
    elif n_inputs == 3:
        train_inputs = np.arange(18, dtype=float).reshape(6, 3)
    else:
        train_inputs = np.arange(12, dtype=float).reshape(6, 2)
    train_outputs = np.sum(train_inputs, axis=1)
    test_inputs = train_inputs[:3]
    test_outputs = np.sum(test_inputs, axis=1)
    return (
        {"inputs": train_inputs, "outputs": train_outputs},
        {"inputs": test_inputs, "outputs": test_outputs},
    )


class FakeTorchTensor:
    def __init__(self, array):
        self.array = np.array(array, dtype=float)
        self.device = "cpu"

    def cuda(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def flatten(self, start_dim=1):
        if self.array.ndim <= 1:
            flat = self.array
        else:
            first_dim = int(np.prod(self.array.shape[:start_dim]))
            second_dim = int(np.prod(self.array.shape[start_dim:]))
            flat = self.array.reshape(first_dim, second_dim)
        return FakeTorchTensor(flat)

    def mean(self):
        return FakeTorchTensor(np.array(self.array.mean()))

    def __sub__(self, other):
        other_array = other.array if isinstance(other, FakeTorchTensor) else other
        return FakeTorchTensor(self.array - other_array)


class FakeEagerTensor:
    def __init__(self, raw):
        self.raw = raw


class TestAdversarialAttackTaskExtended:
    def test_code_execution_error_is_reported(self):
        task = AdversarialAttackTask(use_mock=False)

        result = task.evaluate_code("def draw_proposals(\n")

        assert result.valid is False
        assert "Code execution error" in result.additional_info["error"]

    def test_attack_evaluation_exception_is_reported(self, monkeypatch):
        task = AdversarialAttackTask(use_mock=False, model=object(), test_loader=[])
        monkeypatch.setattr(task, "_evaluate_attack", lambda func: (_ for _ in ()).throw(RuntimeError("boom")))

        result = task.evaluate_code(
            "def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):\n"
            "    return org_img\n"
        )

        assert result.valid is False
        assert "Attack evaluation error" in result.additional_info["error"]

    def test_evaluate_attack_returns_none_when_no_distances_are_collected(self, monkeypatch):
        task = AdversarialAttackTask(use_mock=False, model=object(), test_loader=[], n_test_samples=1)
        fake_ep = types.SimpleNamespace(
            astensor=lambda tensor: FakeEagerTensor(tensor),
            clip=lambda tensor, min_value, max_value: tensor,
        )
        fake_fb = types.SimpleNamespace(
            PyTorchModel=lambda model, bounds: types.SimpleNamespace(bounds=bounds),
            criteria=types.SimpleNamespace(Misclassification=lambda y: ("criterion", y)),
        )
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            linalg=types.SimpleNamespace(norm=lambda value, axis=1: FakeTorchTensor(np.array([1.0]))),
        )
        monkeypatch.setitem(sys.modules, "eagerpy", fake_ep)
        monkeypatch.setitem(sys.modules, "foolbox", fake_fb)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        evo_attack_module = importlib.import_module("evotoolkit.task.python_task.adversarial_attack.evo_attack")
        monkeypatch.setattr(evo_attack_module, "EvoAttack", lambda *args, **kwargs: types.SimpleNamespace(run=lambda *a, **k: None))

        result = task._evaluate_attack(lambda *args: args[0])

        assert result is None

    def test_evaluate_attack_handles_invalid_distances_and_failures(self, monkeypatch):
        samples = [
            (FakeTorchTensor(np.ones((1, 3, 2, 2))), FakeTorchTensor(np.array([1.0]))),
            (FakeTorchTensor(np.ones((1, 3, 2, 2)) * 2), FakeTorchTensor(np.array([0.0]))),
        ]
        task = AdversarialAttackTask(use_mock=False, model=types.SimpleNamespace(cuda=lambda: None), test_loader=samples, n_test_samples=2)

        class FakeAttack:
            def __init__(self):
                self.calls = 0

            def run(self, model, x, criterion):
                self.calls += 1
                if self.calls == 1:
                    return FakeTorchTensor(np.ones((1, 3, 2, 2)) * 5)
                raise RuntimeError("attack failure")

        fake_attack = FakeAttack()
        fake_ep = types.SimpleNamespace(
            astensor=lambda tensor: FakeEagerTensor(tensor),
            clip=lambda tensor, min_value, max_value: tensor,
        )
        fake_fb = types.SimpleNamespace(
            PyTorchModel=lambda model, bounds: types.SimpleNamespace(bounds=bounds),
            criteria=types.SimpleNamespace(Misclassification=lambda y: ("criterion", y)),
        )
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            linalg=types.SimpleNamespace(
                norm=lambda value, axis=1: FakeTorchTensor(np.array([np.nan if fake_attack.calls == 1 else 2.0]))
            ),
        )
        monkeypatch.setitem(sys.modules, "eagerpy", fake_ep)
        monkeypatch.setitem(sys.modules, "foolbox", fake_fb)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        evo_attack_module = importlib.import_module("evotoolkit.task.python_task.adversarial_attack.evo_attack")
        monkeypatch.setattr(evo_attack_module, "EvoAttack", lambda *args, **kwargs: fake_attack)

        result = task._evaluate_attack(lambda *args: args[0])

        assert result == 10.0

    def test_evo_attack_import_error_and_piecewise_scaling(self):
        evo_attack_module = importlib.import_module("evotoolkit.task.python_task.adversarial_attack.evo_attack")

        if not evo_attack_module.FOOLBOX_AVAILABLE:
            with pytest.raises(ImportError):
                evo_attack_module.EvoAttack(types.SimpleNamespace(draw_proposals=lambda *args: args[0]))

        attack = object.__new__(evo_attack_module.EvoAttack)
        values = attack._f_p(np.array([0.0, 0.25, 1.0]))

        assert values.tolist() == [0.5, 1.0, 1.5]


class TestScientificRegressionTaskExtended:
    def test_load_dataset_wraps_download_error(self, monkeypatch):
        original_load_dataset = ScientificRegressionTask._load_dataset
        train_data, test_data = _datasets_with_n_inputs(2)
        monkeypatch.setattr(ScientificRegressionTask, "_load_dataset", lambda self, dataset_name, data_dir: (train_data, test_data))
        task = ScientificRegressionTask(dataset_name="oscillator1")

        class FakeDownloadError(Exception):
            pass

        fake_data_module = types.SimpleNamespace(
            DownloadError=FakeDownloadError,
            get_dataset_path=lambda category, data_dir=None: (_ for _ in ()).throw(FakeDownloadError("network down")),
        )
        monkeypatch.setitem(sys.modules, "evotoolkit.data", fake_data_module)

        with pytest.raises(FileNotFoundError, match="Failed to download dataset"):
            original_load_dataset(task, "oscillator1", None)

    def test_load_dataset_raises_when_downloaded_dataset_is_missing(self, monkeypatch, tmp_path):
        original_load_dataset = ScientificRegressionTask._load_dataset
        train_data, test_data = _datasets_with_n_inputs(2)
        monkeypatch.setattr(ScientificRegressionTask, "_load_dataset", lambda self, dataset_name, data_dir: (train_data, test_data))
        task = ScientificRegressionTask(dataset_name="oscillator1")
        base_dir = tmp_path / "scientific_regression"
        base_dir.mkdir()
        fake_data_module = types.SimpleNamespace(
            DownloadError=RuntimeError,
            get_dataset_path=lambda category, data_dir=None: base_dir,
        )
        monkeypatch.setitem(sys.modules, "evotoolkit.data", fake_data_module)

        with pytest.raises(FileNotFoundError, match="not found after download"):
            original_load_dataset(task, "oscillator1", None)

    def test_evaluate_code_reports_optimization_failures_and_eval_exceptions(self, monkeypatch):
        datasets = _datasets_with_n_inputs(2)
        monkeypatch.setattr(ScientificRegressionTask, "_load_dataset", lambda self, dataset_name, data_dir: datasets)
        task = ScientificRegressionTask(dataset_name="oscillator1", max_params=3)

        monkeypatch.setattr(task, "_evaluate_equation", lambda func, inputs, outputs: (None, ["warn"]))
        result = task.evaluate_code("def equation(x, v, params):\n    return x\n")
        assert result.valid is False
        assert result.additional_info["warnings"] == ["warn"]

        monkeypatch.setattr(task, "_evaluate_equation", lambda func, inputs, outputs: (_ for _ in ()).throw(RuntimeError("calc failed")))
        result = task.evaluate_code("def equation(x, v, params):\n    return x\n")
        assert result.valid is False
        assert "Evaluation error" in result.additional_info["error"]

    def test_evaluate_equation_handles_generic_inputs_nan_and_exceptions(self, monkeypatch):
        datasets = _datasets_with_n_inputs(3)
        monkeypatch.setattr(ScientificRegressionTask, "_load_dataset", lambda self, dataset_name, data_dir: datasets)
        task = ScientificRegressionTask(dataset_name="oscillator1", max_params=2)

        class Result:
            def __init__(self, fun):
                self.fun = fun

        fake_optimize = types.SimpleNamespace(minimize=lambda *args, **kwargs: Result(np.nan))
        monkeypatch.setitem(sys.modules, "scipy.optimize", fake_optimize)

        score, warnings_list = task._evaluate_equation(lambda *args: np.sum(args[0]), datasets[0]["inputs"], datasets[0]["outputs"])
        assert score is None
        assert warnings_list == []

        fake_optimize = types.SimpleNamespace(minimize=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("opt failed")))
        monkeypatch.setitem(sys.modules, "scipy.optimize", fake_optimize)
        score, warnings_list = task._evaluate_equation(lambda *args: np.sum(args[0]), datasets[0]["inputs"], datasets[0]["outputs"])
        assert score is None
        assert warnings_list == []

    def test_get_base_task_description_and_init_solution_for_four_inputs(self, monkeypatch):
        datasets = _datasets_with_n_inputs(4)
        monkeypatch.setattr(ScientificRegressionTask, "_load_dataset", lambda self, dataset_name, data_dir: datasets)
        task = ScientificRegressionTask(dataset_name="bactgrow", max_params=5)

        description = task.get_base_task_description()
        init_solution = task.make_init_sol_wo_other_info()

        assert "temp: np.ndarray" in description
        assert init_solution.evaluation_res is not None
