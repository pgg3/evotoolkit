# Software Improvements Relative to Prior Work

This document is intended for reviewers of the EvoToolkit software package.

The underlying application ideas used in this repository were previously presented in domain papers such as EvoEngineer, CoEvo, and L-AutoDA. Those papers focused on individual research results in specific domains. The `evotoolkit` package contributes a reusable software system that generalizes and productizes those ideas.

## What Is New in EvoToolkit

- A unified `Method -> Interface -> Task` abstraction that decouples evolutionary search, LLM interaction, and domain evaluation.
- A shared top-level API, `evotoolkit.solve(...)`, that dispatches algorithms from interface types instead of requiring application-specific runners.
- Reusable implementations of three LLM-guided evolutionary algorithms in one package: EoH, EvoEngineer, and FunSearch.
- A common output/history format for run state, checkpointing, summaries, and experiment artifacts.
- A pip-installable Python package with task-specific optional extras.
- Public bilingual documentation covering installation, tutorials, API reference, and contributor workflows.
- A cross-platform CI workflow spanning Linux, macOS, and Windows on Python 3.10-3.12.
- An automated test suite that covers the portable CPU-reviewed portion of the package and integration paths built around mocked LLM interactions.

## How This Differs from the Prior Papers

### EvoEngineer

- The EvoEngineer paper is centered on CUDA kernel optimization results.
- EvoToolkit extracts the method into a reusable library component and exposes it through shared task and interface abstractions.
- The package adds packaging, documentation, tests, and a top-level API that were not the focus of the paper.

### CoEvo

- The CoEvo paper focuses on scientific equation discovery.
- EvoToolkit recasts that workflow as a task implementation inside the same framework used by unrelated domains.
- Dataset management, task reuse, and software packaging are standardized across the library rather than being application-specific.

### L-AutoDA

- L-AutoDA studies adversarial attack evolution as a domain application.
- EvoToolkit reuses the same method/task composition model for adversarial attacks, symbolic regression, prompts, and other domains.
- The package adds framework-level documentation, testing, and public distribution practices absent from the standalone research artifact.

## Reviewer-Oriented Summary

Reviewers should treat EvoToolkit as a software unification and engineering contribution built on previously published application ideas. The novelty of this submission is not a new single-domain result; it is the reusable framework, packaging, documentation, and software engineering work that makes those ideas portable across domains.
