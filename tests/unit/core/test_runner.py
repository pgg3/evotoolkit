import json
import threading
from types import SimpleNamespace

import pytest

from evotoolkit.core import EvaluationResult, Instruction, Proposal, ProposalContext, Runner, TaskSpec

SPEC = TaskSpec("count", "Return a number", "string", "answer.txt")


class Archive:
    def __init__(self):
        self.solutions = []

    def __len__(self):
        return len(self.solutions)

    def best(self):
        valid = [s for s in self.solutions if s.score is not None]
        return max(valid, key=lambda s: s.score) if valid else None


class Strategy:
    name = "test"

    def __init__(self, count=1):
        self.archive = Archive()
        self.count = count

    def next_batch(self, step, max_n, remaining_budget, rng):
        return [
            ProposalContext(SPEC, self.archive.solutions[-1:], Instruction("next", 1, "Generate", "code_block"), num_proposals=self.count) for _ in range(max_n)
        ]

    def observe(self, solutions):
        self.archive.solutions.extend(solutions)

    def should_stop(self, step):
        return False


class Proposer:
    name = "script"

    def propose(self, ctx):
        return [Proposal(call_id=ctx.call_id, files={"answer.txt": str(i)}, usage={"tokens": 3}) for i in range(ctx.num_proposals)]


def make_runner(tmp_path, **kwargs):
    defaults = dict(
        evaluator=SimpleNamespace(spec=SPEC, evaluate=lambda files: EvaluationResult(True, float(files["answer.txt"]), "")),
        proposer=Proposer(),
        strategy=Strategy(),
        output_dir=tmp_path,
        budget=7,
        verbose=False,
    )
    return Runner(**(defaults | kwargs))


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("mode", ["generational", "steady"])
def test_exact_budget_and_log_before_observe(tmp_path, mode):
    strategy = Strategy(count=3)
    original = strategy.observe

    def observe(solutions):
        logged = [e["solution"]["id"] for e in events(tmp_path) if e["type"] == "solution"]
        assert all(s.id in logged for s in solutions)
        original(solutions)

    strategy.observe = observe
    runner = make_runner(tmp_path, strategy=strategy, mode=mode)
    assert runner.run().score == 2
    assert runner.samples == 7
    assert runner.status == "completed"
    assert len([e for e in events(tmp_path) if e["type"] == "solution"]) == 7
    seen = set()
    for event in events(tmp_path):
        if event["type"] == "solution":
            assert set(event["solution"]["parent_ids"]) <= seen
            seen.add(event["solution"]["id"])


@pytest.mark.parametrize("mode", ["generational", "steady"])
@pytest.mark.parametrize("failure", ["raise", "empty", "overflow"])
def test_failures_are_bounded_without_spending_budget(tmp_path, mode, failure):
    def propose(ctx):
        if failure == "raise":
            raise RuntimeError("offline")
        return [] if failure == "empty" else [Proposal(ctx.call_id, {})] * (ctx.num_proposals + 1)

    runner = make_runner(tmp_path, proposer=SimpleNamespace(name="broken", propose=propose), max_calls=5, mode=mode)
    assert runner.run() is None
    assert (runner.samples, runner.calls, runner.failed_calls, runner.status) == (0, 5, 5, "max_calls")


@pytest.mark.parametrize("result", [None, EvaluationResult(True, None, ""), EvaluationResult(True, float("nan"), "")])
def test_bad_evaluator_is_logged_as_invalid(tmp_path, result):
    evaluator = SimpleNamespace(spec=SPEC, evaluate=lambda _: result)
    runner = make_runner(tmp_path, evaluator=evaluator, budget=1)
    assert runner.run() is None
    solution = next(e["solution"] for e in events(tmp_path) if e["type"] == "solution")
    assert solution["evaluation"]["valid"] is False
    assert solution["evaluation"]["feedback"]


def test_evaluation_starts_before_slow_sibling_proposal_finishes(tmp_path):
    evaluation_started = threading.Event()
    lock = threading.Lock()
    calls = 0

    def propose(ctx):
        nonlocal calls
        with lock:
            calls += 1
            index = calls
        if index == 2:
            assert evaluation_started.wait(3), "Evaluation waited for all proposal calls"
        return [Proposal(ctx.call_id, {"answer.txt": "1"})]

    def evaluate(files):
        evaluation_started.set()
        return EvaluationResult(True, 1, "")

    runner = make_runner(
        tmp_path,
        budget=2,
        propose_workers=2,
        eval_workers=1,
        proposer=SimpleNamespace(name="pipeline", propose=propose),
        evaluator=SimpleNamespace(spec=SPEC, evaluate=evaluate),
    )
    runner.run()
    assert runner.failed_calls == 0


def test_steady_refills_before_slow_sibling_finishes(tmp_path):
    third_started = threading.Event()
    lock = threading.Lock()
    calls = 0

    def propose(ctx):
        nonlocal calls
        with lock:
            calls += 1
            index = calls
        if index == 2:
            assert third_started.wait(3), "Steady mode waited for batch completion"
        if index == 3:
            third_started.set()
        return [Proposal(ctx.call_id, {"answer.txt": "1"})]

    runner = make_runner(tmp_path, budget=3, propose_workers=2, mode="steady", proposer=SimpleNamespace(name="pipeline", propose=propose))
    runner.run()
    assert runner.failed_calls == 0


def test_resume_restores_counters_archive_rng_and_configuration(tmp_path):
    runner = make_runner(tmp_path, propose_workers=2, seed=19)
    runner.step()
    expected_ids = [s.id for s in runner.strategy.archive.solutions]
    expected_rng = runner.rng.bit_generator.state
    runner.close()
    resumed = Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=Strategy())
    assert (resumed.samples, resumed.calls, resumed.steps, resumed.propose_workers) == (2, 2, 1, 2)
    assert [s.id for s in resumed.strategy.archive.solutions] == expected_ids
    assert resumed.rng.bit_generator.state == expected_rng
    resumed.run()
    assert resumed.samples == 7
    assert len([e for e in events(tmp_path) if e["type"] == "run_start"]) == 1


def test_resume_counts_call_interrupted_before_result(tmp_path):
    runner = make_runner(tmp_path, max_calls=1)
    runner.eventlog.call_started(step=0, call_id="interrupted", instruction="next")
    runner.close()
    resumed = Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=Strategy())
    resumed.run()
    assert resumed.calls == 1
    assert resumed.status == "max_calls"


def test_resume_keeps_solution_before_checkpoint(tmp_path):
    runner = make_runner(tmp_path, propose_workers=1)
    runner.step()
    runner.close()
    path = tmp_path / "events.jsonl"
    rows = path.read_text().splitlines(keepends=True)
    path.write_text("".join(rows[:-1]), encoding="utf-8")
    resumed = Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=Strategy())
    assert resumed.samples == 1 and resumed.steps == 1
    resumed.run()
    assert resumed.samples == 7


@pytest.mark.parametrize("mode", ["generational", "steady"])
def test_resume_partial_work_restores_rng_and_persists_observe_boundary(tmp_path, mode):
    class RandomStrategy(Strategy):
        def __init__(self):
            super().__init__()
            self.batches = []

        def next_batch(self, step, max_n, remaining_budget, rng):
            rng.integers(1000)
            return super().next_batch(step, max_n, remaining_budget, rng)

        def observe(self, solutions):
            self.batches.append([solution.id for solution in solutions])
            super().observe(solutions)

    runner = make_runner(tmp_path, strategy=RandomStrategy(), propose_workers=1, mode=mode, seed=18)
    runner.step()
    expected_rng = runner.rng.bit_generator.state
    runner.close()
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
    resumed = Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=RandomStrategy())
    assert (resumed.steps, resumed.samples, resumed.calls) == (1, 1, 1)
    assert resumed.rng.bit_generator.state == expected_rng
    resumed.step()
    expected_batches = resumed.strategy.batches
    resumed.close()
    with Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=RandomStrategy()) as again:
        assert again.strategy.batches == expected_batches
        assert (again.steps, again.samples, again.calls) == (2, 2, 2)


def test_resume_keeps_rng_for_call_interrupted_before_solution(tmp_path):
    runner = make_runner(tmp_path)
    runner.rng.integers(1000)
    expected_rng = runner.rng.bit_generator.state
    runner.eventlog.call_started(step=0, call_id="interrupted", instruction="next", rng_state=expected_rng)
    runner.close()
    with Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=Strategy()) as resumed:
        assert resumed.rng.bit_generator.state == expected_rng
        assert (resumed.steps, resumed.samples, resumed.calls) == (1, 0, 1)


@pytest.mark.parametrize("mode", ["generational", "steady"])
def test_stops_when_strategy_has_no_more_contexts(tmp_path, mode):
    strategy = Strategy()
    batches = iter([strategy.next_batch(0, 1, 1, None)])
    strategy.next_batch = lambda *args: next(batches, [])
    runner = make_runner(tmp_path, strategy=strategy, propose_workers=1, mode=mode)
    runner.run()
    assert (runner.samples, runner.calls, runner.status) == (1, 1, "stopped")


@pytest.mark.parametrize("mode", ["generational", "steady"])
def test_partial_proposer_batches_release_unused_budget_reservations(tmp_path, mode):
    proposer = SimpleNamespace(name="partial", propose=lambda ctx: [Proposal(ctx.call_id, {"answer.txt": "1"})])
    runner = make_runner(tmp_path, strategy=Strategy(count=3), proposer=proposer, mode=mode, budget=7)
    runner.run()
    assert (runner.samples, runner.calls, runner.failed_calls, runner.status) == (7, 7, 0, "completed")


def test_refuses_overwriting_existing_run(tmp_path):
    make_runner(tmp_path).close()
    with pytest.raises(ValueError, match="resume"):
        make_runner(tmp_path)


def test_resume_rejects_unknown_log_version(tmp_path):
    runner = make_runner(tmp_path)
    runner.close()
    path = tmp_path / "events.jsonl"
    path.write_text(path.read_text().replace('"v": 1', '"v": 99'), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=Strategy())


@pytest.mark.parametrize("kwargs", [{"budget": -1}, {"propose_workers": 0}, {"eval_workers": 0}, {"max_calls": 0}, {"mode": "other"}])
def test_validates_configuration(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_runner(tmp_path, **kwargs)


def test_call_chain_shares_one_id_across_start_result_and_failure(tmp_path):
    class Flaky(Proposer):
        def propose(self, ctx):
            if ctx.call_id.endswith(("0", "2", "4", "6", "8", "a", "c", "e")):
                raise ValueError("boom")
            return super().propose(ctx)

    runner = make_runner(tmp_path, proposer=Flaky(), budget=3, max_calls=12, propose_workers=1)
    runner.run()
    started = {e["call_id"] for e in events(tmp_path) if e["type"] == "call_started"}
    finished = {e["call_id"] for e in events(tmp_path) if e["type"] == "call_failed"} | {
        e["solution"]["call_id"] for e in events(tmp_path) if e["type"] == "solution"
    }
    assert started == finished and len(started) == runner.calls
    assert all(s.call_id in started for s in runner.strategy.archive.solutions)


def test_rejects_proposal_that_does_not_echo_call_id(tmp_path):
    proposer = SimpleNamespace(name="rogue", propose=lambda ctx: [Proposal("mine", {"answer.txt": "1"})])
    runner = make_runner(tmp_path, proposer=proposer, budget=1, max_calls=1)
    runner.run()
    assert runner.failed_calls == 1 and runner.samples == 0


def test_archive_feedback_matches_log_and_resume(tmp_path):
    long = "测" * 5000
    evaluator = SimpleNamespace(spec=SPEC, evaluate=lambda files: EvaluationResult(True, 1.0, long))
    runner = make_runner(tmp_path, evaluator=evaluator, budget=1, max_calls=1)
    runner.run()
    live = runner.strategy.archive.solutions[0].evaluation.feedback
    runner.close()
    resumed = Runner.resume(tmp_path, evaluator=evaluator, proposer=Proposer(), strategy=Strategy())
    assert 0 < len(live.encode("utf-8")) <= 4096
    assert resumed.strategy.archive.solutions[0].evaluation.feedback == live


def test_resume_rejects_changed_strategy_hyperparameters(tmp_path):
    strategy = Strategy()
    strategy.config = {"pop_size": 2}
    runner = make_runner(tmp_path, strategy=strategy, budget=1)
    runner.run()
    runner.close()
    other = Strategy()
    other.config = {"pop_size": 5}
    with pytest.raises(ValueError, match="Strategy configuration"):
        Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=other)
    same = Strategy()
    same.config = {"pop_size": 2}
    assert Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=same).samples == 1


def test_call_failed_uses_runner_call_id_even_if_exception_carries_another(tmp_path):
    def propose(ctx):
        exc = ValueError("backend down")
        exc.call_id = "req-from-backend"
        raise exc

    runner = make_runner(tmp_path, proposer=SimpleNamespace(name="rogue", propose=propose), budget=1, max_calls=1)
    runner.run()
    started = [e["call_id"] for e in events(tmp_path) if e["type"] == "call_started"]
    failed = [e["call_id"] for e in events(tmp_path) if e["type"] == "call_failed"]
    assert started == failed and failed != ["req-from-backend"]


def test_steady_step_observes_one_solution_at_a_time_like_resume(tmp_path):
    class Grouping(Strategy):
        def __init__(self):
            super().__init__()
            self.groups = []

        def observe(self, solutions):
            self.groups.append(len(solutions))
            super().observe(solutions)

    runner = make_runner(tmp_path, strategy=Grouping(), mode="steady", propose_workers=4, budget=8)
    runner.step()
    assert runner.strategy.groups and set(runner.strategy.groups) == {1}
    runner.close()
    replayed = Grouping()
    Runner.resume(tmp_path, evaluator=runner.evaluator, proposer=Proposer(), strategy=replayed)
    assert replayed.groups == runner.strategy.groups
