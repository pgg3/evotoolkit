# Installation

Install the latest stable core SDK:

```bash
pip install evotoolkit
```

This is enough if you want to:

- build your own `PythonTask` or `StringTask`
- run the built-in methods on your own objectives
- use EvoToolkit as a runtime dependency in another package

For local development inside the repository:

```bash
uv sync --group dev
```

The core package does not ship application-specific task dependencies. If you build on EvoToolkit, declare those dependencies in your own package.
