# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for BaseConfig."""

from evotoolkit.core import BaseConfig


class TestBaseConfig:
    def test_create_basic(self, eoh_interface, tmp_output):
        config = BaseConfig(interface=eoh_interface, output_path=tmp_output)
        assert config.interface is eoh_interface
        assert config.output_path == tmp_output
        assert config.verbose is True

    def test_verbose_default_true(self, eoh_interface, tmp_output):
        config = BaseConfig(interface=eoh_interface, output_path=tmp_output)
        assert config.verbose is True

    def test_verbose_false(self, eoh_interface, tmp_output):
        config = BaseConfig(interface=eoh_interface, output_path=tmp_output, verbose=False)
        assert config.verbose is False

    def test_task_property(self, eoh_interface, tmp_output, minimal_task):
        config = BaseConfig(interface=eoh_interface, output_path=tmp_output)
        # task accessed through interface
        assert config.task is minimal_task

    def test_interface_stored(self, eoh_interface, tmp_output):
        config = BaseConfig(interface=eoh_interface, output_path=tmp_output)
        assert config.interface is eoh_interface
