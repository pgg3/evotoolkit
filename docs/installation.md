# Installation

Python 3.10 or newer is required.

```bash
pip install evotoolkit
```

To work from a checkout instead:

```bash
uv sync --group dev
uv run pytest tests/
```

The core dependency is NumPy. Chat models use any client exposing get_response(messages); the included HttpsApi speaks the chat-completions wire format. Model access requires your own endpoint and credentials.

AgentProposer additionally requires a separate headless-cli installation and authenticated backend. See [Extensions](extensions.md). No agent dependency is installed by importing EvoToolkit.
