# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from dataclasses import dataclass

from evotoolkit.core import PopulationMethodState


@dataclass
class EvoEngineerState(PopulationMethodState):
    """Runtime state for EvoEngineer."""
