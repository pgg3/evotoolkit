import numpy as np
import pytest

from evotoolkit.core import EvaluationResult, Proposal, Runner, TaskSpec
from evotoolkit.strategies import EoHStrategy, EvoEngineerStrategy, FunSearchStrategy
from tests.unit.test_archives import solution

SPEC = TaskSpec("test", "Write a solution", "python", "solution.py")
STRATEGIES = [EoHStrategy, EvoEngineerStrategy, FunSearchStrategy]


@pytest.mark.parametrize("factory", STRATEGIES)
def test_bootstrap_respects_slots_and_budget(factory):
    strategy = factory(SPEC)
    contexts = strategy.next_batch(0, 5, 2, np.random.default_rng(0))
    assert len(contexts) == 2
    assert all(not context.parents and context.num_proposals == 1 for context in contexts)
    assert all(context.spec is SPEC for context in contexts)
    assert strategy.next_batch(0, 0, 20, np.random.default_rng(0)) == []
    assert strategy.next_batch(0, 5, 0, np.random.default_rng(0)) == []
    assert not strategy.should_stop(10)


@pytest.mark.parametrize("factory", STRATEGIES)
def test_observe_and_next_batch_are_replayable(factory):
    original, replayed = factory(SPEC), factory(SPEC)
    solutions = [solution(str(index), index) for index in range(7)]
    original.observe(solutions)
    for item in solutions:
        replayed.observe([item])
    first = original.next_batch(5, 4, 4, np.random.default_rng(3))
    assert first == replayed.next_batch(5, 4, 4, np.random.default_rng(3))
    assert first == original.next_batch(5, 4, 4, np.random.default_rng(3))
    assert all(context.parents and context.best.score == 6 for context in first)


def test_eoh_uses_every_operator_with_one_worker():
    strategy = EoHStrategy(SPEC, pop_size=5)
    strategy.observe([solution("a", 1), solution("b", 2)])
    contexts = [strategy.next_batch(step, 1, 10, np.random.default_rng(step))[0] for step in range(4)]
    assert [context.instruction.name for context in contexts] == ["e1", "e2", "m1", "m2"]
    assert [len(context.parents) for context in contexts] == [2, 2, 1, 1]


@pytest.mark.parametrize("factory", [EoHStrategy, EvoEngineerStrategy])
@pytest.mark.parametrize("mode", ["generational", "steady"])
def test_duplicate_proposals_do_not_trap_strategy_in_initialization(factory, mode, tmp_path):
    class Evaluator:
        spec = SPEC

        def evaluate(self, files):
            return EvaluationResult(True, 1, "")

    class RepeatingProposer:
        name = "repeat"

        def __init__(self):
            self.contexts = []

        def propose(self, ctx):
            self.contexts.append(ctx)
            return [Proposal(ctx.call_id, {"solution.py": "same content"})]

    proposer = RepeatingProposer()
    strategy = factory(SPEC)
    runner = Runner(
        evaluator=Evaluator(), proposer=proposer, strategy=strategy, output_dir=tmp_path, budget=6, propose_workers=1, eval_workers=1, mode=mode, verbose=False
    )
    runner.run()
    assert runner.samples == 6
    assert len(strategy.archive) == 1
    assert proposer.contexts[0].instruction.name in {"i1", "init"}
    for context in proposer.contexts:
        if context.parents:
            assert context.instruction.name in {"m1", "m2", "mutation"}
    contexts = strategy.next_batch(runner.steps, 4, 4, np.random.default_rng(0))
    assert all(context.instruction.name in {"m1", "m2", "mutation"} for context in contexts)


def test_evoengineer_provides_best_and_rationale_hints():
    strategy = EvoEngineerStrategy(SPEC)
    strategy.observe([solution(str(index), index) for index in range(5)])
    contexts = strategy.next_batch(2, 2, 2, np.random.default_rng(0))
    assert [context.instruction.name for context in contexts] == ["crossover", "mutation"]
    assert all(context.best.score == 4 and len(context.hints) == 3 for context in contexts)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (EoHStrategy, {"pop_size": 1, "selection_num": 2}),
        (EvoEngineerStrategy, {"pop_size": 1}),
        (FunSearchStrategy, {"num_parents": 0}),
    ],
)
def test_invalid_strategy_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(SPEC, **kwargs)
