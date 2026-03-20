# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from dataclasses import dataclass

from evotoolkit.core import PopulationState


@dataclass
class EvoEngineerState(PopulationState):
    """Runtime state for EvoEngineer."""
