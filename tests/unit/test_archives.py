import numpy as np
import pytest

from evotoolkit.archives import IslandArchive, RankedPopulation
from evotoolkit.core import EvaluationResult, Solution


def solution(content, score, *, valid=True):
    return Solution.create(
        files={"solution.py": content},
        name=content,
        rationale=f"idea {content}",
        parent_ids=[],
        instruction="init",
        proposer="test",
        call_id=content,
        usage={},
        created_step=0,
        evaluation=EvaluationResult(valid, score, ""),
    )


@pytest.mark.parametrize("factory", [RankedPopulation, IslandArchive])
def test_archives_filter_invalid_and_duplicate_content(factory):
    archive = factory()
    for score in (None, float("nan"), float("inf"), -float("inf")):
        archive.add(solution(str(score), score))
    archive.add(solution("invalid", 100, valid=False))
    assert len(archive) == 0
    assert archive.best() is None
    assert archive.sample(2, rng=np.random.default_rng(0)) == []
    archive.add(solution("good", 2))
    archive.add(solution("good", 20))
    assert len(archive) == 1
    assert archive.best().score == 2


def test_ranked_population_keeps_best_and_samples_reproducibly():
    archive = RankedPopulation(2)
    for content, score in (("low", -3), ("high", 5), ("mid", 1)):
        archive.add(solution(content, score))
    assert [item.score for item in archive.solutions] == [5, 1]
    first = archive.sample(2, rng=np.random.default_rng(19))
    second = archive.sample(2, rng=np.random.default_rng(19))
    assert first == second
    rng = np.random.default_rng(0)
    winners = sum(archive.sample(1, rng=rng)[0].score == 5 for _ in range(1000))
    assert 550 < winners < 650


def test_islands_cluster_scores_and_order_parents():
    archive = IslandArchive(1, cluster_sampling_temperature_period=3)
    for content, score in (("first", -1e100), ("second", 1e100), ("same score", 1e100)):
        archive.add(solution(content, score))
    parents = archive.sample(3, rng=np.random.default_rng(7))
    assert [parent.score for parent in parents] == [-1e100, 1e100]
    assert len({parent.id for parent in parents}) == 2


def test_island_resets_replay_without_rng_or_clock():
    original = IslandArchive(3, reset_interval=3)
    replayed = IslandArchive(3, reset_interval=3)
    for index in range(12):
        item = solution(str(index), index)
        original.add(item)
        replayed.add(Solution.from_dict(item.to_dict()))
    assert original.islands == replayed.islands
    assert original.best() == replayed.best()
    assert original.best().score == 11
    assert original.sample(2, rng=np.random.default_rng(4)) == replayed.sample(2, rng=np.random.default_rng(4))
    assert len(original) <= 12


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (RankedPopulation, {"max_size": 0}),
        (RankedPopulation, {"max_size": 1.5}),
        (IslandArchive, {"num_islands": 0}),
        (IslandArchive, {"reset_interval": 0}),
        (IslandArchive, {"cluster_sampling_temperature_init": float("nan")}),
        (IslandArchive, {"cluster_sampling_temperature_period": 0}),
    ],
)
def test_invalid_archive_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
