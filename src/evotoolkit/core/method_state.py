# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from dataclasses import dataclass, field

from .solution import Solution


@dataclass
class MethodState:
    """Serializable runtime state shared by all methods."""

    task_info: dict = field(default_factory=dict)
    sol_history: list[Solution] = field(default_factory=list)
    usage_history: dict[str, list[dict]] = field(default_factory=lambda: {"sample": []})
    status: str = "created"
    bootstrapped: bool = False


@dataclass
class PopulationMethodState(MethodState):
    """Common runtime state for population-based evolutionary methods."""

    generation: int = 0
    tot_sample_nums: int = 0
    population: list[Solution] = field(default_factory=list)
    current_generation_solutions: list[Solution] = field(default_factory=list)
    current_generation_usage: list[dict] = field(default_factory=list)
    best_per_generation: list[dict] = field(default_factory=list)
