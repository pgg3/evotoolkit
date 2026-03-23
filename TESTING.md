# Testing

The core SDK should be testable without any external task package installed.

## Core Test Suite

```bash
uv run pytest
```

This suite validates:

- runtime state, checkpointing, and persistence
- built-in methods: `EoH`, `EvoEngineer`, and `FunSearch`
- generic Python and string task SDK layers
- generic Python and string interfaces
- top-level package exports and `HttpsApi`

## Coverage

```bash
uv run pytest tests/ --cov --cov-config=coverage-full.ini --cov-report=term-missing -m "not cuda and not llm and not slow"
```

## Docs And Packaging

```bash
uv run mkdocs build
uv build --out-dir dist
uvx twine check dist/*
```
