"""Append-only JSONL events and complete proposer traces."""

import json
import os
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Iterator

from .types import Solution, TaskSpec, _plain


class EventLog:
    VERSION = 1
    FEEDBACK_LIMIT = 4096

    def __init__(self, output_dir: str | os.PathLike):
        self.output_dir = Path(output_dir)
        self.traces_dir = self.output_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "events.jsonl"
        self._lock = Lock()
        self._truncate_partial_tail()
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")

    def _truncate_partial_tail(self) -> None:
        """Drop an unterminated final line left by an interrupted write; complete lines are never touched."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        head, _, tail = data.rpartition(b"\n")
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            with self.path.open("r+b") as stream:
                stream.truncate(len(head) + 1 if head else 0)
        else:
            with self.path.open("ab") as stream:
                stream.write(b"\n")

    def _write(self, event_type: str, **payload) -> None:
        event = {"v": self.VERSION, "type": event_type, "ts": datetime.now(timezone.utc).isoformat(), **payload}
        # shortcut: non-finite floats use Python JSON extensions; map them when strict external readers are required.
        line = json.dumps(_plain(event), ensure_ascii=False) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()

    def run_start(self, *, task: TaskSpec, strategy: str, proposer: str, seed: int, budget: int, config: dict | None = None) -> None:
        try:
            package_version = version("evotoolkit")
        except PackageNotFoundError:
            package_version = "unknown"
        self._write(
            "run_start",
            task=task.to_dict(),
            strategy=strategy,
            proposer=proposer,
            seed=seed,
            budget=budget,
            env={"python": platform.python_version(), "platform": platform.system(), "evotoolkit": package_version},
            **({"config": config} if config is not None else {}),
        )

    def solution(self, *, step: int, solution: Solution) -> None:
        payload = solution.to_dict()
        evaluation = payload["evaluation"]
        if evaluation is not None:
            feedback = evaluation["feedback"].encode("utf-8")
            if len(feedback) > self.FEEDBACK_LIMIT:
                evaluation["feedback"] = feedback[: self.FEEDBACK_LIMIT].decode("utf-8", errors="ignore")
                evaluation["feedback_truncated"] = True
        self._write("solution", step=step, solution=payload)

    def call_started(self, *, step: int, call_id: str, instruction: str, rng_state: dict | None = None) -> None:
        self._write("call_started", step=step, call_id=call_id, instruction=instruction, **({"rng_state": rng_state} if rng_state is not None else {}))

    def call_failed(self, *, step: int, call_id: str, instruction: str, error: str) -> None:
        self._write("call_failed", step=step, call_id=call_id, instruction=instruction, error=error)

    def step_end(
        self,
        *,
        step: int,
        archive_size: int,
        best_id: str | None,
        best_score: float | None,
        proposals: int,
        valid: int,
        calls: int,
        failed_calls: int,
        rng_state: dict | None = None,
    ) -> None:
        self._write(
            "step_end",
            step=step,
            archive_size=archive_size,
            best_id=best_id,
            best_score=best_score,
            proposals=proposals,
            valid=valid,
            calls=calls,
            failed_calls=failed_calls,
            **({"rng_state": rng_state} if rng_state is not None else {}),
        )

    def run_end(self, *, status: str, steps: int, samples: int) -> None:
        self._write("run_end", status=status, steps=steps, samples=samples)

    def trace(self, call_id: str, payload: dict) -> None:
        if not isinstance(call_id, str) or re.fullmatch(r"[a-zA-Z0-9_-]+", call_id) is None:
            raise ValueError("call_id must contain only letters, digits, underscores, or hyphens")
        path = self.traces_dir / f"{call_id}.json"
        if path.resolve().parent != self.traces_dir.resolve():
            raise ValueError("trace path escapes traces directory")
        content = json.dumps(_plain(payload), ensure_ascii=False)
        with self._lock:
            path.write_text(content, encoding="utf-8")

    def events(self) -> Iterator[dict]:
        with self.path.open("rb") as stream:
            for number, line in enumerate(stream, start=1):
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeDecodeError) as error:
                    if not line.endswith(b"\n"):
                        return  # a write in progress or an interrupted process; the next open truncates it
                    raise ValueError(f"Invalid event at line {number}") from error
                if not isinstance(event, dict):
                    raise ValueError(f"Invalid event at line {number}: expected an object")
                yield event

    def close(self) -> None:
        with self._lock:
            self._stream.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
