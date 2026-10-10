"""The shared, proposal-budgeted search loop."""

import math
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from .eventlog import EventLog
from .types import EvaluationResult, Proposal, Solution


class Runner:
    def __init__(
        self, *, evaluator, proposer, strategy, output_dir, budget, max_calls=None, propose_workers=4, eval_workers=4, mode="generational", seed=0, verbose=True
    ):
        self._configure(evaluator, proposer, strategy, output_dir, budget, max_calls, propose_workers, eval_workers, mode, seed, verbose)
        path = Path(output_dir) / "events.jsonl"
        if path.exists() and path.stat().st_size:
            raise ValueError("Output directory contains a run; use Runner.resume() or a new directory")
        self.eventlog = EventLog(output_dir)
        self.eventlog.run_start(task=evaluator.spec, strategy=strategy.name, proposer=proposer.name, seed=seed, budget=budget, config=self.config)
        self._attach_log()

    def _configure(self, evaluator, proposer, strategy, output_dir, budget, max_calls, propose_workers, eval_workers, mode, seed, verbose):
        for name, value, minimum in (("budget", budget, 0), ("propose_workers", propose_workers, 1), ("eval_workers", eval_workers, 1)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if max_calls is None:
            max_calls = max(1, budget * 3)
        if not isinstance(max_calls, int) or isinstance(max_calls, bool) or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if mode not in {"generational", "steady"}:
            raise ValueError("mode must be 'generational' or 'steady'")
        if len(strategy.archive):
            raise ValueError("Pass a fresh strategy with an empty archive")
        self.evaluator, self.proposer, self.strategy = evaluator, proposer, strategy
        self.output_dir = Path(output_dir)
        self.budget, self.max_calls = budget, max_calls
        self.propose_workers, self.eval_workers = propose_workers, eval_workers
        self.mode, self.seed, self.verbose = mode, seed, verbose
        self.rng = np.random.default_rng(seed)
        self.steps = self.samples = self.calls = self.failed_calls = 0
        self.status = "running"
        self.config = dict(
            max_calls=max_calls,
            propose_workers=propose_workers,
            eval_workers=eval_workers,
            mode=mode,
            verbose=verbose,
            strategy=getattr(strategy, "config", {}),
        )

    def _attach_log(self):
        if hasattr(self.proposer, "eventlog"):
            self.proposer.eventlog = self.eventlog

    @property
    def best_solution(self):
        return self.strategy.archive.best()

    def _stopped(self):
        return self.samples >= self.budget or self.calls >= self.max_calls or self.strategy.should_stop(self.steps)

    def _evaluate(self, proposal):
        start = time.perf_counter()
        try:
            result = self.evaluator.evaluate(proposal.files)
            if not isinstance(result, EvaluationResult):
                raise TypeError("Evaluator must return EvaluationResult")
            if result.valid and (result.score is None or not math.isfinite(result.score)):
                result = replace(result, valid=False, feedback="Valid evaluation requires a finite score")
        except Exception as exc:
            result = EvaluationResult(valid=False, score=None, feedback=f"{type(exc).__name__}: {exc}")
        # Truncate once here so archive, prompts and the event log all see the same feedback (resume must be a function of the log).
        feedback = result.feedback.encode("utf-8")[: EventLog.FEEDBACK_LIMIT].decode("utf-8", errors="ignore")
        return replace(result, feedback=feedback, cost_seconds=time.perf_counter() - start)

    def _contexts(self, max_n, remaining):
        contexts = self.strategy.next_batch(self.steps, max_n, remaining, self.rng)
        if len(contexts) > max_n:
            raise ValueError("Strategy returned more contexts than max_n")
        accepted = []
        for ctx in contexts:
            if not isinstance(ctx.num_proposals, int) or isinstance(ctx.num_proposals, bool) or ctx.num_proposals < 1:
                raise ValueError("num_proposals must be a positive integer")
            if remaining <= 0:
                break
            ctx = replace(ctx, num_proposals=min(ctx.num_proposals, remaining))
            accepted.append(ctx)
            remaining -= ctx.num_proposals
        return accepted

    def _pipeline(self, *, steady):
        proposing, evaluating = {}, {}
        reserved = 0
        observed = []
        had_results = False
        before = (self.samples, self.calls, self.failed_calls)
        exhausted = False
        with ThreadPoolExecutor(self.propose_workers) as proposals, ThreadPoolExecutor(self.eval_workers) as evaluations:
            while True:
                free = min(self.propose_workers - len(proposing), self.max_calls - self.calls)
                remaining = self.budget - self.samples - reserved
                if not exhausted and free > 0 and remaining > 0 and not self.strategy.should_stop(self.steps):
                    ctxs = self._contexts(1 if steady else free, remaining)
                    if not ctxs:
                        exhausted = True
                    for ctx in ctxs:
                        call_id = uuid4().hex
                        self.eventlog.call_started(step=self.steps, call_id=call_id, instruction=ctx.instruction.name, rng_state=self.rng.bit_generator.state)
                        self.calls += 1
                        reserved += ctx.num_proposals
                        ctx = replace(ctx, call_id=call_id)
                        proposing[proposals.submit(self.proposer.propose, ctx)] = (ctx, call_id)
                    if not steady:
                        exhausted = True
                    elif ctxs and len(proposing) < self.propose_workers:
                        continue
                pending = set(proposing) | set(evaluating)
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in proposing:
                        ctx, call_id = proposing.pop(future)
                        try:
                            batch = future.result()
                            if not isinstance(batch, list) or len(batch) > ctx.num_proposals:
                                raise ValueError("Proposer must return a list within num_proposals")
                            for proposal in batch:
                                if (
                                    not isinstance(proposal, Proposal)
                                    or proposal.call_id != call_id
                                    or not isinstance(proposal.files, dict)
                                    or any(not isinstance(k, str) or not isinstance(v, str) for k, v in proposal.files.items())
                                ):
                                    raise ValueError("Proposer returned an invalid Proposal")
                            if not batch:
                                raise ValueError("Proposer returned no proposals")
                        except Exception as exc:
                            reserved -= ctx.num_proposals
                            self.failed_calls += 1
                            self.eventlog.call_failed(
                                step=self.steps,
                                call_id=call_id,
                                instruction=ctx.instruction.name,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            continue
                        reserved -= ctx.num_proposals - len(batch)
                        for proposal in batch:
                            evaluating[evaluations.submit(self._evaluate, proposal)] = (ctx, proposal)
                    else:
                        ctx, proposal = evaluating.pop(future)
                        solution = Solution.create(
                            files=proposal.files,
                            name=proposal.name,
                            rationale=proposal.rationale,
                            parent_ids=[parent.id for parent in ctx.parents],
                            instruction=ctx.instruction.name,
                            proposer=self.proposer.name,
                            call_id=proposal.call_id,
                            usage=proposal.usage,
                            created_step=self.steps,
                            evaluation=future.result(),
                        )
                        self.eventlog.solution(step=self.steps, solution=solution)
                        self.samples += 1
                        reserved -= 1
                        had_results = True
                        if steady:
                            self._observe([solution])
                            self._checkpoint([solution], before)
                            before = (self.samples, self.calls, self.failed_calls)
                            exhausted = False
                        else:
                            observed.append(solution)
                if steady and not proposing and not evaluating and self._stopped():
                    break
        if not steady:
            self._observe(observed)
            self._checkpoint(observed, before)
        elif before != (self.samples, self.calls, self.failed_calls):
            self._checkpoint([], before)
        if not had_results and self.calls == before[1] and exhausted:
            self.status = "stopped"

    def _observe(self, solutions):
        # A generational batch is a set: order it by id so archives that depend on insertion order
        # (island seeding, ties) do not depend on thread completion order, in runs and in replay alike.
        # Steady mode observes one solution at a time everywhere: run(), a bounded step(), and replay.
        if self.mode == "generational":
            self.strategy.observe(sorted(solutions, key=lambda s: s.id))
        else:
            for solution in solutions:
                self.strategy.observe([solution])

    def _checkpoint(self, solutions, before):
        best = self.best_solution
        self.eventlog.step_end(
            step=self.steps,
            archive_size=len(self.strategy.archive),
            best_id=best.id if best else None,
            best_score=best.score if best else None,
            proposals=self.samples - before[0],
            valid=sum(s.score is not None for s in solutions),
            calls=self.calls - before[1],
            failed_calls=self.failed_calls - before[2],
            rng_state=self.rng.bit_generator.state,
        )
        self.steps += 1
        if self.verbose:
            print(f"Step {self.steps}: {self.samples}/{self.budget} proposals, {self.calls}/{self.max_calls} calls, best={best.score if best else None}")

    def step(self):
        """Run one bounded batch; run() continuously refills workers in steady mode."""
        if self.status == "running" and not self._stopped():
            self._pipeline(steady=False)

    def run(self):
        try:
            if self.mode == "steady" and self.status == "running" and not self._stopped():
                self._pipeline(steady=True)
            else:
                while self.status == "running" and not self._stopped():
                    self.step()
            if self.status == "running":
                self.status = "completed" if self.samples >= self.budget else "max_calls" if self.calls >= self.max_calls else "stopped"
            self.eventlog.run_end(status=self.status, steps=self.steps, samples=self.samples)
            return self.best_solution
        except BaseException:
            self.status = "interrupted"
            self.eventlog.run_end(status=self.status, steps=self.steps, samples=self.samples)
            raise
        finally:
            self.close()

    def close(self):
        self.eventlog.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @classmethod
    def resume(cls, output_dir, *, evaluator, proposer, strategy):
        path = Path(output_dir) / "events.jsonl"
        if not path.exists():
            raise ValueError("No event log to resume")
        log = EventLog(output_dir)
        try:
            events = log.events()
            start = next(events, None)
            if start is None or start.get("type") != "run_start":
                raise ValueError("Event log has no run_start")
            if start.get("v") != EventLog.VERSION:
                raise ValueError("Unsupported event log version")
            if start["task"] != evaluator.spec.to_dict() or start["strategy"] != strategy.name or start["proposer"] != proposer.name:
                raise ValueError("Task, strategy and proposer must match the saved run")
            config = start.get("config", {})
            if config.get("strategy", {}) != getattr(strategy, "config", {}):
                raise ValueError(f"Strategy configuration must match the saved run: {config.get('strategy', {})}")
            runner = cls.__new__(cls)
            runner._configure(
                evaluator,
                proposer,
                strategy,
                output_dir,
                start["budget"],
                config.get("max_calls"),
                config.get("propose_workers", 4),
                config.get("eval_workers", 4),
                config.get("mode", "generational"),
                start["seed"],
                config.get("verbose", True),
            )
            runner.eventlog = log
            pending = []
            partial = []
            before = (0, 0, 0)
            for event in events:
                if event.get("v") != EventLog.VERSION:
                    raise ValueError("Unsupported event log version")
                kind = event["type"]
                if kind == "call_started":
                    runner.calls += 1
                    if "rng_state" in event:
                        runner.rng.bit_generator.state = event["rng_state"]
                elif kind == "call_failed":
                    runner.failed_calls += 1
                elif kind == "solution":
                    solution = Solution.from_dict(event["solution"])
                    runner.samples += 1
                    pending.append(solution)
                    partial.append(solution)
                    if runner.mode == "steady":
                        runner._observe(pending)
                        pending = []
                elif kind == "step_end":
                    if pending:
                        runner._observe(pending)
                        pending = []
                    runner.steps = event["step"] + 1
                    if "rng_state" in event:
                        runner.rng.bit_generator.state = event["rng_state"]
                    partial = []
                    before = (runner.samples, runner.calls, runner.failed_calls)
            if pending:
                runner._observe(pending)
            if before != (runner.samples, runner.calls, runner.failed_calls):
                # Persist the recovered boundary so subsequent resumes replay the same observe batches.
                runner.steps = max([runner.steps, *(s.created_step for s in partial)])
                runner._checkpoint(partial, before)
            runner._attach_log()
            return runner
        except BaseException:
            log.close()
            raise
