import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["generational", "steady"])
def test_offline_example(tmp_path, mode):
    path = Path(__file__).parents[2] / "examples/custom_task/my_custom_task.py"
    spec = importlib.util.spec_from_file_location("custom_task", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    best = module.run_example(output_dir=tmp_path, offline=True, mode=mode)
    assert best.score == 0
    assert best.evaluation.metrics["mse"] == 0
    assert module.run_example(output_dir=tmp_path, offline=True, resume=True).id == best.id
