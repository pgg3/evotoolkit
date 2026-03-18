# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for FunSearchConfig."""

from evotoolkit.evo_method.funsearch.run_config import FunSearchConfig


class TestFunSearchConfig:
    def test_default_values(self, funsearch_interface, tmp_output):
        cfg = FunSearchConfig(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=None,
            verbose=False,
        )
        assert cfg.max_sample_nums == 45
        assert cfg.num_islands == 5
        assert cfg.max_population_size == 1000
        assert cfg.num_samplers == 5
        assert cfg.num_evaluators == 5
        assert cfg.programs_per_prompt == 2

    def test_custom_values(self, funsearch_interface, tmp_output):
        cfg = FunSearchConfig(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=None,
            max_sample_nums=100,
            num_islands=3,
            max_population_size=500,
            verbose=False,
        )
        assert cfg.max_sample_nums == 100
        assert cfg.num_islands == 3
        assert cfg.max_population_size == 500

    def test_task_accessible(self, funsearch_interface, tmp_output):
        cfg = FunSearchConfig(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=None,
            verbose=False,
        )
        assert cfg.task is not None

    def test_interface_stored(self, funsearch_interface, tmp_output):
        cfg = FunSearchConfig(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=None,
            verbose=False,
        )
        assert cfg.interface is funsearch_interface

    def test_ignores_extra_kwargs(self, funsearch_interface, tmp_output, capsys):
        # Should not raise, just silently ignore
        cfg = FunSearchConfig(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=None,
            max_generations=10,  # ignored kwarg
            pop_size=20,  # ignored kwarg
            verbose=True,
        )
        assert cfg.max_sample_nums == 45
        captured = capsys.readouterr()
        assert "Ignoring" in captured.out

    def test_output_path_stored(self, funsearch_interface, tmp_output):
        cfg = FunSearchConfig(
            interface=funsearch_interface,
            output_path=tmp_output,
            running_llm=None,
            verbose=False,
        )
        assert cfg.output_path == tmp_output
