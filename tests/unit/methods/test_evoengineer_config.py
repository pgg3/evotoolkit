# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EvoEngineerInterface and EvoEngineerConfig."""

import pytest

from evotoolkit.core.operator import Operator
from evotoolkit.evo_method.evoengineer import EvoEngineerConfig


class TestEvoEngineerInterface:
    def test_make_init_sol_has_name_baseline(self, evoengineer_interface):
        sol = evoengineer_interface.make_init_sol()
        assert sol.other_info["name"] == "Baseline"
        assert sol.other_info["thought"] == "Baseline"

    def test_get_init_operators_returns_list(self, evoengineer_interface):
        ops = evoengineer_interface.get_init_operators()
        assert isinstance(ops, list)
        assert len(ops) > 0

    def test_get_offspring_operators_returns_list(self, evoengineer_interface):
        ops = evoengineer_interface.get_offspring_operators()
        assert isinstance(ops, list)
        assert len(ops) > 0

    def test_init_operators_selection_size_zero(self, evoengineer_interface):
        ops = evoengineer_interface.get_init_operators()
        for op in ops:
            assert op.selection_size == 0, f"Init operator '{op.name}' must have selection_size=0"

    def test_valid_require_attribute(self, evoengineer_interface):
        assert hasattr(evoengineer_interface, "valid_require")
        assert evoengineer_interface.valid_require == 2

    def test_get_operator_prompt_returns_messages(self, evoengineer_interface, valid_solution):
        prompt = evoengineer_interface.get_operator_prompt(
            "crossover",
            selected_individuals=[valid_solution],
            current_best_sol=valid_solution,
            random_thoughts=[],
        )
        assert isinstance(prompt, list)
        assert len(prompt) > 0


class TestEvoEngineerConfig:
    def test_create_basic(self, evoengineer_interface, tmp_output):
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        config = EvoEngineerConfig(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=mock_llm,
            verbose=False,
        )
        assert config.max_generations == 10
        assert config.pop_size == 5
        assert config.running_llm is mock_llm

    def test_config_gets_operators_from_interface(self, evoengineer_interface, tmp_output):
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        config = EvoEngineerConfig(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=mock_llm,
        )
        assert len(config.init_operators) > 0
        assert len(config.offspring_operators) > 0

    def test_config_raises_when_no_init_operators(self, minimal_task, tmp_output):
        from unittest.mock import MagicMock

        from evotoolkit.core.method_interface import EvoEngineerInterface
        from evotoolkit.core.solution import Solution

        class NoInitInterface(EvoEngineerInterface):
            def get_init_operators(self):
                return []  # Invalid: empty

            def get_offspring_operators(self):
                return [Operator("mutate", selection_size=1)]

            def get_operator_prompt(self, operator_name, selected_individuals, current_best_sol, random_thoughts, **kwargs):
                return [{"role": "user", "content": "test"}]

            def parse_response(self, response_str):
                return Solution("code")

        iface = NoInitInterface(minimal_task)
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="at least one init operator"):
            EvoEngineerConfig(interface=iface, output_path=tmp_output, running_llm=mock_llm)

    def test_config_raises_when_init_op_nonzero_selection(self, minimal_task, tmp_output):
        from unittest.mock import MagicMock

        from evotoolkit.core.method_interface import EvoEngineerInterface
        from evotoolkit.core.solution import Solution

        class BadInitInterface(EvoEngineerInterface):
            def get_init_operators(self):
                return [Operator("bad_init", selection_size=2)]  # Invalid

            def get_offspring_operators(self):
                return [Operator("mutate", selection_size=1)]

            def get_operator_prompt(self, operator_name, selected_individuals, current_best_sol, random_thoughts, **kwargs):
                return [{"role": "user", "content": "test"}]

            def parse_response(self, response_str):
                return Solution("code")

        iface = BadInitInterface(minimal_task)
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="selection_size=0"):
            EvoEngineerConfig(interface=iface, output_path=tmp_output, running_llm=mock_llm)
