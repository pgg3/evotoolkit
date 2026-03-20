# Changelog

All notable user-facing changes to EvoToolkit are documented here.

## [3.0.0] - 2026-03-20

### Changed

- Rebuilt the core runtime around explicit algorithm objects instead of `evotoolkit.solve(...)`.
- Replaced the old config plus run-state split with method-owned dataclass state and a thin `Method` lifecycle.
- Switched checkpointing to `checkpoint/state.pkl` plus `checkpoint/manifest.json`, while keeping `history/` and `summary/` as readable artifacts only.

### Added

- `MethodState` and per-algorithm state dataclasses.
- `RunStore` as the unified persistence layer for checkpoint, history, and summary artifacts.
- Explicit resume flow through method re-instantiation plus `load_checkpoint()`.

### Removed

- `BaseConfig`
- `BaseRunStateDict`
- `HistoryManager`
- `evotoolkit.solve(...)`
- implicit algorithm inference and config-class assembly
- `run_state.json`
- standalone `programs_database.json` checkpointing for FunSearch

## [2.0.0] - 2026-03-20

### Changed

- Repositioned `evotoolkit` as a core SDK instead of a monolithic framework-plus-task distribution.
- Removed concrete task families, hardware workflows, and bundled reproducibility assets from the core public surface.
- Simplified the active documentation and testing story around the reusable engine itself.

### Added

- Companion package split via `evotoolkit-tasks`.
- Explicit core-vs-companion package boundary across packaging, docs, and examples.
- Minimal core-first testing path based only on generic Python and String task abstractions.

### Removed

- Built-in scientific regression, prompt optimization, adversarial attack, control, CUDA, and CANN task families from the core distribution.
- MLOSS-specific reviewer-facing positioning from the active release narrative.

## [1.0.0] - 2026-03-19

### Notes

- `1.0.0` was the last pre-split release that shipped both the reusable framework and the built-in task families together.
