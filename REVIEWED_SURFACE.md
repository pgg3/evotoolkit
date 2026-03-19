# Reviewed Surface For The MLOSS Submission

This document defines the primary reviewer-facing software surface for EvoToolkit.

## Core Framework

The primary reviewed surface includes the reusable framework itself:

- `src/evotoolkit/core/**`
- `src/evotoolkit/__init__.py`
- `src/evotoolkit/registry.py`
- `src/evotoolkit/tools/llm.py`
- `src/evotoolkit/data/**`
- `src/evotoolkit/evo_method/eoh/**`
- `src/evotoolkit/evo_method/evoengineer/**`
- `src/evotoolkit/evo_method/funsearch/**`

## Reference Task Layers

The primary reviewed surface also includes CPU-reviewable reference-task layers that demonstrate portability:

- `src/evotoolkit/task/python_task/**`
- `src/evotoolkit/task/string_optimization/**`
- `src/evotoolkit/task/cuda_engineering/__init__.py`
- `src/evotoolkit/task/cuda_engineering/cuda_task.py`
- `src/evotoolkit/task/cuda_engineering/method_interface/**`

These modules are treated as reference adapters rather than the product definition of EvoToolkit.

## Explicitly Excluded From The Primary Reviewed Surface

The following areas remain in the repository but are not part of the primary reviewed surface:

- `examples/**` tutorial and reproducibility scripts
- `src/evotoolkit/evo_method/cann_initer/**`
- `src/evotoolkit/task/cann_init/**`
- `src/evotoolkit/task/cuda_engineering/ai_cuda_engineer/**`
- `src/evotoolkit/task/cuda_engineering/evaluator/**`

These excluded areas are retained as either:

- experimental adjacent workflows (`cann_init`)
- hardware-backed runtime helpers (`cuda_engineering/evaluator/**`, `ai_cuda_engineer/**`)
- tutorial assets (`examples/**`)

## Coverage Interpretation

- The primary MLOSS coverage metric is computed with `coverage-reviewed.ini`.
- The whole-repository transparency audit is computed with `coverage-full.ini`.
- Reviewers should treat the primary metric as the quality signal for the reusable framework and its CPU-reviewable reference-task layers.
