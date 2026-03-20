# Archived Reviewed-Surface Note

This file is retained only as a historical pointer from the discontinued JMLR MLOSS submission workflow.

The active `2.0.0` packaging model is simpler:

- `evotoolkit` is the core SDK
- `evotoolkit-tasks` is the companion package for concrete domains and hardware-backed workflows

The current core package boundary is:

- `src/evotoolkit/core/**`
- `src/evotoolkit/__init__.py`
- `src/evotoolkit/registry.py`
- `src/evotoolkit/tools/**`
- `src/evotoolkit/evo_method/eoh/**`
- `src/evotoolkit/evo_method/evoengineer/**`
- `src/evotoolkit/evo_method/funsearch/**`
- `src/evotoolkit/task/python_task/python_task.py`
- `src/evotoolkit/task/python_task/method_interface/**`
- `src/evotoolkit/task/string_optimization/string_task.py`
- `src/evotoolkit/task/string_optimization/method_interface/**`

Everything else should be treated as companion-package material or legacy source retained only until the repository cleanup is fully physical.
