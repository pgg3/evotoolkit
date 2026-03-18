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

1. Create a subclass of `BaseTask` (for Python tasks use `PythonTask`; for string tasks use `StringTask`)
2. Implement required methods: `evaluate_solution()`, `get_base_task_description()`, `make_init_sol_wo_other_info()`
3. Create a corresponding `MethodInterface` subclass (e.g., `EoHPythonInterface`, `EvoEngineerPythonInterface`)
4. Register the task with `@register_task`
5. Add tests in `tests/unit/tasks/` and update documentation

## Adding a New Algorithm

1. Create a subclass of `Method`
2. Register with `@register_algorithm`
3. Implement `run()` and `_get_run_state_class()`
4. Create a corresponding `Config` class (subclass of `BaseConfig`)
5. Add tests in `tests/unit/methods/`

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
