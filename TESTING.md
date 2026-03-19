# Testing And Coverage

EvoToolkit maintains three complementary validation views.

## 1. Routine CI Subset

This is the portable CPU-only subset used for routine cross-platform validation:

```bash
uv run pytest tests/ -m "not cuda and not llm and not slow"
```

## 2. Primary Reviewed Surface

This is the reviewer-facing coverage metric for the MLOSS submission. It measures the core framework plus CPU-reviewable reference-task layers defined in [`REVIEWED_SURFACE.md`](REVIEWED_SURFACE.md).

```bash
uv run pytest tests/ --cov --cov-config=coverage-reviewed.ini --cov-report=term-missing -m "not cuda and not llm and not slow"
```

This primary metric excludes:

- `examples/**`
- experimental CANN modules
- hardware-runtime CUDA compilation / benchmarking helpers

## 3. Full-Package Audit

This stricter audit is retained for transparency across the entire source tree, including optional and hardware-coupled modules:

```bash
uv run pytest tests/ --cov --cov-config=coverage-full.ini --cov-report=term -m "not cuda and not llm and not slow"
```

The full-package audit is not the headline MLOSS quality metric. It is a transparency view for the whole repository.

## Packaging And Docs Checks

```bash
uv run mkdocs build
uv build --out-dir dist
uvx twine check dist/*
```
