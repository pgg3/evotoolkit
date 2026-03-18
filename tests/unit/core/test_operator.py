# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for Operator data class."""

from evotoolkit.core.operator import Operator


class TestOperator:
    def test_create_with_name_only(self):
        op = Operator("mutate")
        assert op.name == "mutate"
        assert op.selection_size == 0

    def test_create_with_selection_size(self):
        op = Operator("crossover", selection_size=2)
        assert op.name == "crossover"
        assert op.selection_size == 2

    def test_create_init_operator(self):
        op = Operator("init", selection_size=0)
        assert op.selection_size == 0

    def test_name_stored(self):
        op = Operator("E1_crossover")
        assert op.name == "E1_crossover"

    def test_large_selection_size(self):
        op = Operator("tournament", selection_size=10)
        assert op.selection_size == 10
