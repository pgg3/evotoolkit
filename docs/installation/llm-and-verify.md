# LLM Setup & Verify

Configure your LLM API and verify your installation.

---

## Verify Installation

```python
import evotoolkit
from evotoolkit.core import Solution

print(f"EvoToolkit version: {evotoolkit.__version__}")
print("✅ Installation successful!")
```

---

## LLM API Setup

EvoToolkit uses an external LLM API (e.g., OpenAI GPT-4). Configure credentials in code:

```python
from evotoolkit.tools import HttpsApi

llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",  # Your API endpoint
    key="your-api-key-here",  # Your API key
    model="gpt-4o"
)
```

You can obtain a key from OpenAI or use other OpenAI-compatible services.

