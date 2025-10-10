# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Base configuration class for evolutionary methods.

This module contains the base configuration class that is shared
across all evolutionary algorithms in the framework.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evotoolkit.core import BaseTask
    from evotoolkit.core.method_interface import BaseMethodInterface


class BaseConfig:
    """
    Base configuration class for evolutionary methods.

    Note: task is accessed via interface.task to avoid data redundancy.
    """

    def __init__(
        self,
        interface: 'BaseMethodInterface',
        output_path: str,
        verbose: bool = True
    ):
        self.interface = interface
        self.output_path = output_path
        self.verbose = verbose

    @property
    def task(self) -> 'BaseTask':
        """Access task through interface to avoid redundancy."""
        return self.interface.task