# Contributing to EvoToolkit

Thanks for helping improve EvoToolkit.

---

## Ways to Contribute

- Report bugs or unclear behavior
- Improve documentation and examples
- Add tests for uncovered behavior
- Submit focused fixes or new task integrations

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/evotoolkit.git
cd evotoolkit
```

### 2. Install the Development Environment

```bash
uv sync --group dev

# Optional task extras
uv sync --extra scientific_regression
uv sync --extra prompt_engineering
uv sync --extra adversarial_attack
uv sync --extra cuda_engineering
uv sync --extra control_box2d
uv sync --extra cann_init
uv sync --extra all_tasks
```

### 3. Create a Branch

```bash
git checkout -b feature/your-change
```

---

## Code Style

EvoToolkit uses Ruff for linting and formatting.

```bash
uv run ruff format .
uv run ruff check .
```

Please keep new public APIs documented and add tests for bug fixes or behavioral changes.

---

## Validation

Run the portable test subset before opening a pull request:

```bash
uv run pytest tests/ -m "not cuda and not llm and not slow"
uv run mkdocs build
```

If you changed packaging metadata, also verify:

```bash
uv build --out-dir dist
uvx twine check dist/*
```

---

## Pull Requests

1. Keep each PR focused on one change set.
2. Explain the user-facing effect or bug being fixed.
3. Link related issues or discussion threads when available.
4. Update docs/examples if the public behavior changes.

---

## Questions

- GitHub Issues: <https://github.com/pgg3/evotoolkit/issues>
- GitHub Discussions: <https://github.com/pgg3/evotoolkit/discussions>
- Email: `pguo6680@gmail.com`
