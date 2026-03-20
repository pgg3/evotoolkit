# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""EvoToolkit core SDK public entry points."""

# Import algorithms to trigger registration decorators
from evotoolkit.evo_method.eoh import EoH
from evotoolkit.evo_method.evoengineer import EvoEngineer
from evotoolkit.evo_method.funsearch import FunSearch
from evotoolkit.registry import list_algorithms

__author__ = "Ping Guo"

# Read version from package metadata (set in pyproject.toml)
try:
    from importlib.metadata import version

    __version__ = version("evotoolkit")
except Exception:
    # Fallback for development/editable install
    __version__ = "3.0.0.dev0"


# Export public API
__all__ = ["list_algorithms", "__version__", "__author__", "EoH", "EvoEngineer", "FunSearch"]
