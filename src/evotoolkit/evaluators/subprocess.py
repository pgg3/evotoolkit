"""Opt-in process isolation; this is not a security sandbox."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from evotoolkit.core.types import Files

from .files import file_path, write_files


def subprocess_evaluate(files: Files, entry: str | list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    """Run a Python entry file, or an explicit argv, in a temporary directory.

    Return stdout/stderr and returncode (124 for timeout, 1 for setup failure).
    The caller interprets the output and constructs an EvaluationResult.
    """
    command = entry
    try:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with TemporaryDirectory(prefix="evotoolkit-eval-") as directory:
            root = Path(directory)
            write_files(root, files)
            if isinstance(entry, str):
                command = [sys.executable, str(file_path(root, entry))]
            elif not isinstance(entry, list) or not entry or any(not isinstance(arg, str) for arg in entry):
                raise ValueError("entry must be a relative Python file or a nonempty argv list")
            return subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        stdout = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
        return subprocess.CompletedProcess(command, 124, stdout, stderr + f"\nEvaluation timed out after {timeout} seconds")
    except (OSError, ValueError, TypeError) as exc:
        return subprocess.CompletedProcess(command, 1, "", f"{type(exc).__name__}: {exc}")
