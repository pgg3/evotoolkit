# Contributing to EvoToolkit

Thank you for your interest in contributing to EvoToolkit! This guide will help you get started.

## Development Setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/<your-username>/evotoolkit.git
   cd evotoolkit
   ```

2. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync --group dev
   ```

3. **Verify** your setup:
   ```bash
   uv run pytest
   ```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff format .        # Format code
uv run ruff check .         # Lint code
uv run ruff check . --fix   # Auto-fix lint issues
```

- Line length: 160 characters
- Quote style: double quotes
- Import sorting: handled by Ruff (isort rules)

## Running Tests

```bash
uv run pytest                              # Run all tests
uv run pytest --cov --cov-report=html      # With coverage report
uv run pytest -m "not slow"               # Skip slow tests
uv run pytest -m "not cuda"               # Skip CUDA tests
uv run pytest -m "not cuda and not llm"   # CI-compatible subset
```

## Adding a New Task

1. Keep concrete domain tasks outside the core package by default; `src/evotoolkit` should stay focused on reusable SDK surfaces
2. Create a subclass of `Task` (for Python tasks use `PythonTask`; for string tasks use `StringTask`)
3. Build a `TaskSpec` in `build_spec()`, `build_python_spec()`, or `build_string_spec()`
4. Implement `evaluate()` or the task-type-specific evaluation hook
5. Reuse a generic `MethodInterface` when possible; only add a custom interface when the prompt/response contract really differs
6. Register the task with `@register_task` only if you need registry integration; explicit imports are the default workflow
7. Add tests in `tests/unit/tasks/` and update documentation

Do not put initial-solution lifecycle into the task API. If a method needs special bootstrap behavior, keep it inside that method or its prompt design.

## Adding a New Algorithm

1. Prefer subclassing `IterativeMethod` for new step-wise algorithms
2. Register with `@register_algorithm`
3. Implement `step_iteration()` and `should_stop_iteration()`
4. Optionally implement `prepare_initialization()`, `initialize_iteration()`, or a custom `state_cls`
5. Drop down to raw `Method` only if you need a non-standard lifecycle
6. Add tests in `tests/unit/methods/`

`IterativeMethod` already provides:

- default state construction from `task.spec`
- default best-solution selection from `state.sol_history`
- standard checkpoint persistence through `RunStore`

For generation-based population algorithms, `PopulationMethod` is the higher-level base and adds population helpers plus generation artifact flushing.

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Implement** your changes with tests
3. **Ensure** all tests pass: `uv run pytest`
4. **Ensure** code style: `uv run ruff check . && uv run ruff format --check .`
5. **Submit** a pull request with a clear description

### PR Requirements

- All existing tests must pass
- New features must include tests
- Code must pass Ruff linting and formatting checks
- Documentation should be updated if the public API changes

## Reporting Issues

Please use [GitHub Issues](https://github.com/pgg3/evotoolkit/issues) to report bugs or request features. Include:

- **Bug reports**: Python version, OS, minimal reproduction steps, full error traceback
- **Feature requests**: Use case description, proposed API design (if applicable)
