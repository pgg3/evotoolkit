# Changelog

All notable user-facing changes to EvoToolkit are documented here.

## [1.0.1] - 2026-03-23

This stable patch release keeps the runtime API unchanged while republishing the finalized core-only package surface under a fresh PyPI version.

### Highlights

- Stabilized the public runtime around explicit method objects: `EoH`, `EvoEngineer`, and `FunSearch`.
- Reduced the core package to reusable runtime surfaces only: methods, state, persistence, generic tasks, and generic interfaces.
- Finalized the public repository surface down to a strict core-only package layout.

### Packaging

- Published metadata for the stable `1.0.1` release line.
- Kept runtime dependencies minimal: `numpy` and `scipy`.
- Shipped a runnable custom-task example in `examples/custom_task/my_custom_task.py`.

### Documentation

- Rewrote the active package docs around the stable core workflow.
- Reduced the public docs to a small bilingual core set: index, installation, quickstart, extensions, and migration.
- Removed legacy docs and wording that still described deleted APIs, old task families, obsolete dependency models, or prerelease flows.

### Notes

- Standardized task definition around `TaskSpec`, `PythonTask`, and `StringTask`.
- Standardized persistence around `RunStore`, `checkpoint/state.pkl`, and `checkpoint/manifest.json`.
- Removed the old high-level `evotoolkit.solve(...)` entry point from the stable public API.
- Earlier `1.0.0rc*` prereleases existed for validation before the stable surface was locked.

## [1.0.0] - 2026-03-23

This was the first stable release of the standalone EvoToolkit core package.
