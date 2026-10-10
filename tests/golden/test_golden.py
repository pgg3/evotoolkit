# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import json

import pytest

from tests.golden.golden import RUNS, golden_path, produce


@pytest.mark.parametrize("name", sorted(RUNS))
def test_matches_golden(name):
    with open(golden_path(name), encoding="utf-8") as f:
        expected = json.load(f)
    assert produce(name) == expected
