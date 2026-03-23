# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""EvoToolkit core SDK public entry points."""

from evotoolkit.evo_method.eoh import EoH
from evotoolkit.evo_method.evoengineer import EvoEngineer
from evotoolkit.evo_method.funsearch import FunSearch

_BUILTIN_METHODS = (EoH, EvoEngineer, FunSearch)

__author__ = "Ping Guo"

# Read version from package metadata (set in pyproject.toml)
try:
    from importlib.metadata import version

    __version__ = version("evotoolkit")
except Exception:
    # Fallback for development/editable install
    __version__ = "1.0.1rc1.dev0"


def list_algorithms() -> list[str]:
    """Return the built-in algorithm names shipped with EvoToolkit."""

    return [method.algorithm_name for method in _BUILTIN_METHODS]


# Export public API
__all__ = ["list_algorithms", "__version__", "__author__", "EoH", "EvoEngineer", "FunSearch"]
