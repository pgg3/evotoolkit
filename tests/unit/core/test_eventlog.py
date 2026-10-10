import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from evotoolkit.core.eventlog import EventLog
from evotoolkit.core.types import EvaluationResult, Solution, TaskSpec


def _sol(feedback="ok"):
    return Solution.create(
        files={"a.py": "x"},
        name="test",
        rationale="reason",
        parent_ids=[],
        instruction="init",
        proposer="mock",
        call_id="call",
        usage={},
        created_step=0,
        evaluation=EvaluationResult(True, 1.0, feedback),
    )


def test_events_round_trip(tmp_path):
    with EventLog(tmp_path) as log:
        log.run_start(task=TaskSpec("test", "task", "python", "a.py"), strategy="eoh", proposer="mock", seed=0, budget=3, config={"max_calls": 10})
        log.call_started(step=0, call_id="call", instruction="init")
        log.solution(step=0, solution=_sol())
        log.call_failed(step=0, call_id="failed", instruction="init", error="oops")
        log.step_end(step=0, archive_size=1, best_id=_sol().id, best_score=1.0, proposals=1, valid=1, calls=2, failed_calls=1, rng_state={"state": 1})
        log.run_end(status="completed", steps=1, samples=1)
        events = list(log.events())
    assert [event["type"] for event in events] == ["run_start", "call_started", "solution", "call_failed", "step_end", "run_end"]
    assert all(event["v"] == 1 and datetime.fromisoformat(event["ts"]).utcoffset().total_seconds() == 0 for event in events)
    assert events[0]["config"] == {"max_calls": 10}
    assert events[0]["env"].keys() == {"python", "platform", "evotoolkit"}
    assert events[-2]["rng_state"] == {"state": 1}
    assert Solution.from_dict(events[2]["solution"]) == _sol()


def test_feedback_truncation_is_utf8_and_does_not_mutate(tmp_path):
    solution = _sol("测" * 2000)
    with EventLog(tmp_path) as log:
        log.solution(step=0, solution=solution)
        evaluation = next(log.events())["solution"]["evaluation"]
    assert len(evaluation["feedback"].encode("utf-8")) <= 4096
    assert evaluation["feedback_truncated"] is True
    assert solution.evaluation.feedback == "测" * 2000


def test_trace_is_complete_and_path_safe(tmp_path):
    payload = {"response": "测" * 10000}
    with EventLog(tmp_path) as log:
        log.trace("call-1", payload)
        for invalid in ("../escape", "..\\escape", "C:escape", "", "/absolute"):
            with pytest.raises(ValueError):
                log.trace(invalid, payload)
    assert json.loads((tmp_path / "traces" / "call-1.json").read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("tail", [b'{"v": 1, "ty', b'{"feedback": "\xe6\xb5'])
def test_partial_tail_is_skipped_then_truncated_on_reopen(tmp_path, tail):
    with EventLog(tmp_path) as log:
        log.run_end(status="completed", steps=0, samples=0)
        with log.path.open("ab") as stream:
            stream.write(tail)
        assert len(list(log.events())) == 1
    complete = log.path.read_bytes()[: -len(tail)]
    with EventLog(tmp_path) as resumed:
        assert resumed.path.read_bytes() == complete
        resumed.run_end(status="completed", steps=1, samples=1)
        assert [event["type"] for event in resumed.events()] == ["run_end", "run_end"]


@pytest.mark.parametrize("data", [b"{}\nbroken\n{}\n", b"{}\nbroken\n", b"{}\n[]\n"])
def test_corrupt_middle_or_terminated_final_line_raises(tmp_path, data):
    with EventLog(tmp_path) as log:
        log.path.write_bytes(data)
        with pytest.raises(ValueError, match="line 2"):
            list(log.events())


def test_reopen_appends_to_complete_event_without_newline(tmp_path):
    (tmp_path / "events.jsonl").write_bytes(b'{"type":"old"}')
    with EventLog(tmp_path) as log:
        log.run_end(status="completed", steps=1, samples=1)
        assert [event["type"] for event in log.events()] == ["old", "run_end"]


def test_concurrent_traces_and_solutions(tmp_path):
    with EventLog(tmp_path) as log:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(log.solution, step=i, solution=_sol()) for i in range(50)]
            futures += [pool.submit(log.trace, str(i), {"response": str(i)}) for i in range(50)]
            for future in futures:
                future.result()
        assert len(list(log.events())) == 50
    assert len(list((tmp_path / "traces").glob("*.json"))) == 50
    assert all(json.loads(line)["type"] == "solution" for line in log.path.read_text(encoding="utf-8").splitlines())
