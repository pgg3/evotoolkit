# Changelog

All notable user-facing changes to EvoToolkit are documented here.

## [1.0.0] - 2026-03-21

This is the first stable release of the standalone EvoToolkit core package.

### Highlights

- Stabilized the public runtime around explicit method objects: `EoH`, `EvoEngineer`, and `FunSearch`.
- Reduced the core package to reusable runtime surfaces only: methods, state, persistence, generic tasks, and generic interfaces.
- Standardized task definition around `TaskSpec`, `PythonTask`, and `StringTask`.
- Standardized persistence around `RunStore`, `checkpoint/state.pkl`, and `checkpoint/manifest.json`.
- Removed the old high-level `evotoolkit.solve(...)` entry point from the stable public API.

### Packaging

- Published metadata for the first stable `1.0.0` release line.
- Kept runtime dependencies minimal: `numpy` and `scipy`.
- Shipped a runnable custom-task example in `examples/custom_task/my_custom_task.py`.

### Documentation

- Rewrote the active package docs around the stable core workflow.
- Documented the RC-to-`1.0.0` migration path for custom tasks and explicit method instantiation.

### Notes

- Concrete domain tasks are intentionally out of scope for the core package and should live in separate packages or repositories built on top of EvoToolkit.
