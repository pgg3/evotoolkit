# Tools API

Utilities and helpers for EvoToolkit, including LLM API clients.

---

## HttpsApi

See the dedicated page: [HttpsApi](tools/https-api.md).

**Purpose:**

- Connect to LLM chat APIs
- Send prompts and receive responses
- Handle authentication and retries automatically

**Usage Example:**

```python
from evotoolkit.tools import HttpsApi
import os

# Create LLM API client
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o",
    temperature=1.0
)

# Send a prompt
messages = [{"role": "user", "content": "Write a Python function to compute fibonacci"}]
response, usage = llm_api.get_response(messages)

print(response)
print(f"Tokens used: {usage['total_tokens']}")
```

**Using with evotoolkit.solve():**

```python
import evotoolkit
from evotoolkit.task.python_task import ScientificRegressionTask, EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

# Configure LLM
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o"
)

# Create task and interface
task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)

# Solve with the LLM
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

---

### Constructor & Examples

See the [HttpsApi](tools/https-api.md) page for the full constructor, parameters, and usage examples.

---

### Methods

- [get_response()](tools/https-api.md)
- [get_embedding()](tools/https-api.md)

---

## Environment Variables

### OPENAI_API_KEY

Store your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Then use in Python:

```python
import os
from evotoolkit.tools import HttpsApi

llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o"
)
```

### LLM_API_URL / LLM_API_KEY

For custom configurations:

```bash
export LLM_API_URL="https://your-api.com/v1/chat/completions"
export LLM_API_KEY="your-key"
export LLM_MODEL="gpt-4o"
```

---

## Best Practices

### Do's ✅

- Store API keys in environment variables (never hardcode)
- Use appropriate timeout values for your use case
- Handle rate limits gracefully
- Monitor token usage to control costs
- Use lower temperatures (0.0-0.5) for deterministic outputs
- Use higher temperatures (0.7-1.5) for creative outputs

### Don'ts ❌

- Don't commit API keys to git
- Don't use excessively low timeouts (< 30s)
- Don't ignore token usage metrics
- Don't disable retries for production code

---

## Provider-Specific Notes

### OpenAI

```python
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o"
)
```

- **Models**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Rate limits**: Check your account tier
- **Docs**: https://platform.openai.com/docs/api-reference

### Anthropic Claude

Requires a compatible proxy or API gateway:

```python
llm_api = HttpsApi(
    api_url="https://your-gateway.com/v1/chat/completions",
    key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-3-5-sonnet-20241022"
)
```

- **Models**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- **Note**: Requires OpenAI-compatible API format
- **Docs**: https://docs.anthropic.com/

### Custom Providers

Many LLM providers offer OpenAI-compatible APIs:

```python
llm_api = HttpsApi(
    api_url="api.custom-provider.com",  # Hostname only
    key="your-key",
    model="provider-model-name"
)
```

Check your provider's documentation for:
- API endpoint URL
- Authentication method
- Supported models
- Request/response format

---

## Troubleshooting

### Connection Errors

**Problem:** `RuntimeError: Model Response Error!`

**Solutions:**
- Check your API URL is correct
- Verify your API key is valid
- Ensure network connectivity
- Check provider status page

### Timeout Errors

**Problem:** Requests timing out

**Solutions:**
- Increase `timeout` parameter
- Check network latency
- Try a smaller model
- Reduce prompt complexity

### Rate Limiting

**Problem:** Too many requests

**Solutions:**
- Add delays between requests
- Reduce parallelism
- Upgrade API tier
- Implement exponential backoff

---

## Next Steps

- See [Core API](core.md) for using LLMs with `evotoolkit.solve()`
- Check [Methods API](methods.md) for evolutionary algorithms
- Try the [Scientific Regression Tutorial](../tutorials/built-in/scientific-regression.md)

