# Testing and Coverage

EvoToolkit maintains two complementary testing views:

- **Portable reviewed scope**: the CPU-only test suite and default coverage view used in public CI, excluding hardware-coupled CUDA and Ascend modules that are not runnable in standard cross-platform jobs.
- **Full source-tree view**: a stricter audit used to understand how much of the entire reviewed package is exercised, including optional modules that are not always runnable in CI.

## Portable CI Subset

Run the same subset used for routine cross-platform validation:

```bash
uv run pytest tests/ -m "not cuda and not llm and not slow"
```

To measure portable reviewed-scope coverage with the default repository configuration:

```bash
uv run pytest tests/ --cov --cov-report=term -m "not cuda and not llm and not slow"
```

## Full Source-Tree Coverage

To audit coverage across the entire package source tree, use the alternate coverage config:

```bash
uv run pytest tests/ --cov --cov-config=coverage-full.ini --cov-report=term -m "not cuda and not llm and not slow"
```

This view is intentionally stricter than the portable reviewed-scope metric and helps track how much of the reviewed package surface is covered, including optional or hardware-coupled code paths.

## Packaging and Docs Checks

```bash
uv run mkdocs build
uv build --out-dir dist
uvx twine check dist/*
```
