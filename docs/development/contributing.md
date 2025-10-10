# Contributing to EvoToolkit

Thank you for your interest in contributing to EvoToolkit! This guide will help you get started.

---

## Ways to Contribute

- 🐛 **Report bugs** - Submit issues on GitHub
- 💡 **Suggest features** - Share your ideas
- 📝 **Improve documentation** - Fix typos, add examples
- 🔧 **Submit code** - Fix bugs or add features
- 🎓 **Share examples** - Contribute tutorials and use cases

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/evotoolkit.git
cd evotool
```

### 2. Install Development Dependencies

```bash
# Install development dependencies
uv sync --group dev

# Optional: Install specific task dependencies
uv sync --extra cuda_engineering       # For CUDA tasks
uv sync --extra scientific_regression  # For scientific regression
uv sync --extra adversarial_attack     # For adversarial attacks
uv sync --extra all_tasks              # All task dependencies
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Code Style

EvoToolkit uses:
- **Black** for code formatting
- **isort** for import sorting
- **Type hints** throughout the codebase

Format your code:

```bash
uv run black .
uv run isort .
```

---

## Submitting Changes

1. Commit your changes with clear messages
2. Push to your fork
3. Open a Pull Request
4. Respond to review feedback

---

## Questions?

Join [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions) or email pguo6680@gmail.com
