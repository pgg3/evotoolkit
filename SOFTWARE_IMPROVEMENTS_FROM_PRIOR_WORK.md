# Software Improvements Relative to Prior Work

This document is intended for reviewers of the EvoToolkit software package.

## Non-Claims

This MLOSS submission does **not** claim:

- a new theoretical evolutionary algorithm beyond the cited method papers
- a new scientific result replacing EvoEngineer, CoEvo, or L-AutoDA
- that the `examples/` directory is itself the core software contribution

Instead, the submission claims a reusable software framework that packages earlier domain-specific ideas into a shared, reviewable system.

## Framework-Level Additions

| Software Claim | What EvoToolkit Adds | Reviewer-Visible Evidence |
|---|---|---|
| Unified framework | A shared `Method -> Interface -> Task` abstraction spanning multiple LLM-driven evolutionary methods | `src/evotoolkit/core/`, `paper/secs/1_framework.tex`, architecture figure |
| Shared execution substrate | A single top-level `evotoolkit.solve(...)` entry point, algorithm registry, run-state handling, and history output | `src/evotoolkit/__init__.py`, `src/evotoolkit/registry.py`, `src/evotoolkit/core/history_manager.py` |
| Multi-method software release | Reusable implementations of EoH, EvoEngineer, and FunSearch in one package | `src/evotoolkit/evo_method/`, public API docs, package README |
| Reference task adapters | Built-in task families that validate portability across Python, string, control, adversarial, and selected CUDA shells | `src/evotoolkit/task/`, tutorials, API reference |
| Public packaging | Pip-installable package metadata, optional extras, changelog, and release artifacts | `pyproject.toml`, `CHANGELOG.md`, PyPI metadata |
| Reviewable engineering evidence | Automated tests, CI, docs, and explicit reviewed-surface coverage guidance | `tests/`, GitHub Actions, `TESTING.md`, `REVIEWED_SURFACE.md` |
| Bilingual user-facing docs | English and Chinese installation guides, tutorials, and API pages | `docs/`, MkDocs site |

## Relation To Prior Domain Papers

| Prior Artifact | Original Scope | What EvoToolkit Adds As Software |
|---|---|---|
| EvoEngineer | CUDA kernel optimization results and workflow | Extracts the method into a reusable algorithm implementation with shared interfaces, packaging, tests, docs, and top-level dispatch |
| CoEvo | Scientific equation discovery workflow | Recasts the domain pipeline as a reusable reference task inside the same framework used by unrelated domains |
| L-AutoDA | Black-box adversarial attack evolution | Integrates the task as another reference adapter under the same framework and release infrastructure |

## Relation To Reference Tasks And Examples

- `task/**` is part of the package, but these modules are **reference task implementations** rather than the product definition of EvoToolkit.
- `examples/**` contains runnable tutorial and reproducibility scripts. These scripts help users reproduce workflows, but they are not the primary software contribution.
- CUDA remains a hardware-backed reference task family. Its reviewed surface focuses on CPU-reviewable task and interface layers rather than runtime-bound benchmarking helpers.
- CANN is kept in the repository as an experimental adjacent workflow and is not part of the primary reviewed surface for this submission.

## Reviewer Summary

Reviewers should evaluate EvoToolkit as a reusable software framework with shared abstractions, execution infrastructure, packaging, testing, and documentation. The framework is validated through reference task families and case studies derived from prior application domains, but the novelty claim is the software system itself.
