# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EvoEngineerConfig covering uncovered lines (42, 51, 55, 59)."""

import pytest

from evotoolkit.core import Operator


class TestEvoEngineerConfig:
    def test_init_valid(self, evoengineer_interface, tmp_output):
        from evotoolkit.evo_method.evoengineer.run_config import EvoEngineerConfig

        class MockLLM:
            def get_response(self, messages, **kwargs):
                return "code", {}

        config = EvoEngineerConfig(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
        )
        assert config.max_generations == 10
        assert config.max_sample_nums == 45

    def test_init_empty_init_operators_raises(self, minimal_task, tmp_output):
        from evotoolkit.core.method_interface import EvoEngineerInterface
        from evotoolkit.core.solution import Solution
        from evotoolkit.evo_method.evoengineer.run_config import EvoEngineerConfig

        class BadInterface(EvoEngineerInterface):
            def get_init_operators(self):
                return []  # Empty — should raise

            def get_offspring_operators(self):
                return [Operator("cross", 2)]

            def get_operator_prompt(self, op_name, selected, best, thoughts, **kwargs):
                return []

            def parse_response(self, r):
                return Solution(r)

        class MockLLM:
            def get_response(self, messages, **kwargs):
                return "code", {}

        with pytest.raises(ValueError, match="at least one init operator"):
            EvoEngineerConfig(
                interface=BadInterface(minimal_task),
                output_path=tmp_output,
                running_llm=MockLLM(),
            )

    def test_init_empty_offspring_operators_raises(self, minimal_task, tmp_output):
        from evotoolkit.core.method_interface import EvoEngineerInterface
        from evotoolkit.core.solution import Solution
        from evotoolkit.evo_method.evoengineer.run_config import EvoEngineerConfig

        class BadInterface(EvoEngineerInterface):
            def get_init_operators(self):
                return [Operator("init", 0)]

            def get_offspring_operators(self):
                return []  # Empty — should raise

            def get_operator_prompt(self, op_name, selected, best, thoughts, **kwargs):
                return []

            def parse_response(self, r):
                return Solution(r)

        class MockLLM:
            def get_response(self, messages, **kwargs):
                return "code", {}

        with pytest.raises(ValueError, match="at least one offspring operator"):
            EvoEngineerConfig(
                interface=BadInterface(minimal_task),
                output_path=tmp_output,
                running_llm=MockLLM(),
            )

    def test_init_operator_with_wrong_selection_size_raises(self, minimal_task, tmp_output):
        from evotoolkit.core.method_interface import EvoEngineerInterface
        from evotoolkit.core.solution import Solution
        from evotoolkit.evo_method.evoengineer.run_config import EvoEngineerConfig

        class BadInterface(EvoEngineerInterface):
            def get_init_operators(self):
                return [Operator("init", 2)]  # selection_size != 0 — should raise

            def get_offspring_operators(self):
                return [Operator("cross", 2)]

            def get_operator_prompt(self, op_name, selected, best, thoughts, **kwargs):
                return []

            def parse_response(self, r):
                return Solution(r)

        class MockLLM:
            def get_response(self, messages, **kwargs):
                return "code", {}

        with pytest.raises(ValueError, match="selection_size=0"):
            EvoEngineerConfig(
                interface=BadInterface(minimal_task),
                output_path=tmp_output,
                running_llm=MockLLM(),
            )

    def test_get_init_operators(self, evoengineer_interface, tmp_output):
        from evotoolkit.evo_method.evoengineer.run_config import EvoEngineerConfig

        class MockLLM:
            def get_response(self, messages, **kwargs):
                return "code", {}

        config = EvoEngineerConfig(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
        )
        ops = config.get_init_operators()
        assert len(ops) > 0
        assert all(op.selection_size == 0 for op in ops)

    def test_get_offspring_operators(self, evoengineer_interface, tmp_output):
        from evotoolkit.evo_method.evoengineer.run_config import EvoEngineerConfig

        class MockLLM:
            def get_response(self, messages, **kwargs):
                return "code", {}

        config = EvoEngineerConfig(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
        )
        ops = config.get_offspring_operators()
        assert len(ops) > 0

    def test_get_all_operators(self, evoengineer_interface, tmp_output):
        from evotoolkit.evo_method.evoengineer.run_config import EvoEngineerConfig

        class MockLLM:
            def get_response(self, messages, **kwargs):
                return "code", {}

        config = EvoEngineerConfig(
            interface=evoengineer_interface,
            output_path=tmp_output,
            running_llm=MockLLM(),
        )
        all_ops = config.get_all_operators()
        init_ops = config.get_init_operators()
        offspring_ops = config.get_offspring_operators()
        assert len(all_ops) == len(init_ops) + len(offspring_ops)
