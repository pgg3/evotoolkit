# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for EoHConfig."""

from evotoolkit.evo_method.eoh import EoHConfig


class TestEoHConfig:
    def test_create_defaults(self, eoh_interface, tmp_output):
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        config = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=mock_llm,
        )
        assert config.max_generations == 10
        assert config.max_sample_nums == 45
        assert config.pop_size == 5
        assert config.selection_num == 2
        assert config.use_e2_operator is True
        assert config.use_m1_operator is True
        assert config.use_m2_operator is True
        assert config.num_samplers == 5
        assert config.num_evaluators == 5
        assert config.verbose is True

    def test_create_custom_params(self, eoh_interface, tmp_output):
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        config = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=mock_llm,
            max_generations=3,
            pop_size=10,
            verbose=False,
        )
        assert config.max_generations == 3
        assert config.pop_size == 10
        assert config.verbose is False

    def test_task_accessible_via_config(self, eoh_interface, tmp_output, minimal_task):
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        config = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=mock_llm,
        )
        assert config.task is minimal_task

    def test_running_llm_stored(self, eoh_interface, tmp_output):
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        config = EoHConfig(
            interface=eoh_interface,
            output_path=tmp_output,
            running_llm=mock_llm,
        )
        assert config.running_llm is mock_llm
