# Contributing to EvoToolkit

This repository publishes the reusable core SDK. Contributions should keep `src/evotoolkit` focused on generic runtime behavior, not application-specific task packages.

## Development Setup

```bash
git clone https://github.com/<your-username>/evotoolkit.git
cd evotoolkit
uv sync --group dev
```

Verify the environment before opening a change:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mkdocs build
```

## Working on the Core SDK

Task-related contributions should stay within the generic SDK surface:

1. Subclass `PythonTask` or `StringTask`.
2. Return a `TaskSpec` from `build_python_spec()` or `build_string_spec()`.
3. Implement the matching evaluation hook.
4. Reuse a generic interface when possible, or add a custom `MethodInterface` if the prompt/response contract genuinely differs.
5. Add tests in `tests/unit/tasks/` and update the public docs when behavior changes.

Do not add concrete domain datasets, hardware-specific evaluation flows, or application-specific task families to this repository.

## Working on Methods

Algorithm contributions should build on the runtime lifecycle that already exists:

1. Prefer subclassing `IterativeMethod` for step-wise search.
2. Use `PopulationMethod` for generation-based search with population helpers.
3. Drop down to raw `Method` only if you need a non-standard lifecycle.
4. Add tests in `tests/unit/methods/` and checkpoint coverage when persistence behavior changes.

## Documentation Expectations

The public documentation surface is intentionally small and bilingual. If you change public behavior, keep these English and Chinese pages aligned:

- `docs/index*.md`
- `docs/installation*.md`
- `docs/quickstart*.md`
- `docs/extensions*.md`
- `docs/migration*.md`

## Packaging Checks

If you change packaging or release metadata, also verify:

```bash
uv build --out-dir dist
uvx twine check dist/*
```

## Reporting Issues

Please use [GitHub Issues](https://github.com/pgg3/evotoolkit/issues) for bug reports and feature requests. Include Python version, OS, reproduction steps, and the full traceback when applicable.
