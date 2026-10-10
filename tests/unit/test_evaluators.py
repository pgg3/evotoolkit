import os
import sys

import pytest

from evotoolkit.evaluators import subprocess_evaluate
from evotoolkit.evaluators.files import write_files


def test_subprocess_evaluates_nested_files():
    result = subprocess_evaluate({"main.py": "from helper.value import VALUE\nprint(VALUE)", "helper/value.py": "VALUE = 42"}, "main.py")
    assert result.returncode == 0
    assert result.stdout.strip() == "42"


def test_subprocess_reports_errors_and_argv():
    result = subprocess_evaluate({"main.py": "raise RuntimeError('bad candidate')"}, [sys.executable, "main.py"])
    assert result.returncode != 0
    assert "bad candidate" in result.stderr


def test_subprocess_timeout():
    result = subprocess_evaluate({"main.py": "import time\nprint('started', flush=True)\ntime.sleep(10)"}, "main.py", timeout=0.2)
    assert result.returncode == 124
    assert "timed out" in result.stderr


@pytest.mark.parametrize(
    "path", ["../outside.py", "/tmp/outside.py", "C:/outside.py", "a/../../outside.py", "a\\outside.py", "a:stream", "./alias.py", "CON", "NUL/file.py"]
)
def test_subprocess_rejects_unsafe_paths(path):
    result = subprocess_evaluate({path: "print('unsafe')"}, path)
    assert result.returncode == 1
    assert "Unsafe artifact path" in result.stderr


def test_subprocess_missing_command_and_invalid_timeout():
    assert subprocess_evaluate({}, ["evotoolkit-does-not-exist"]).returncode == 1
    assert subprocess_evaluate({}, "main.py", timeout=0).returncode == 1


def test_writer_rejects_symlink_directory(tmp_path):
    root, outside = tmp_path / "workspace", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Filesystem does not allow symlinks: {exc}")
    with pytest.raises(ValueError, match="link"):
        write_files(root, {"link/escaped.py": "x"})
    assert not (outside / "escaped.py").exists()


@pytest.mark.skipif(os.name != "nt", reason="Case aliases apply to Windows")
def test_writer_rejects_windows_case_aliases(tmp_path):
    with pytest.raises(ValueError, match="alias"):
        write_files(tmp_path, {"a.py": "x", "A.py": "y"})
    assert not list(tmp_path.iterdir())
