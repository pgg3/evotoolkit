# Testing

The core SDK should be testable without any concrete task package installed.

## Core Test Suite

```bash
uv run pytest tests/ -m "not cuda and not llm and not slow"
```

This suite validates:

- generic Python task abstractions
- generic String task abstractions
- algorithm execution
- registry behavior
- top-level `solve(...)`

## Coverage

```bash
uv run pytest tests/ --cov --cov-config=coverage-full.ini --cov-report=term-missing -m "not cuda and not llm and not slow"
```

`coverage-reviewed.ini` is kept as a compatibility alias for the same core-only source tree.

## Docs And Packaging

```bash
uv run mkdocs build
uv build --out-dir dist
uvx twine check dist/*
```

## Companion Package

Concrete domain tests now belong in `evotoolkit-tasks` and should be run from the `tasks/` workspace package.
