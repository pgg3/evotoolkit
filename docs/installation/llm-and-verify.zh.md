# LLM 设置与验证

配置 LLM API 并验证安装是否成功。

---

## 验证安装

```python
import evotoolkit
from evotoolkit.core import Solution

print(f"EvoToolkit 版本: {evotoolkit.__version__}")
print("✅ 安装成功！")
```

---

## LLM API 设置

EvoToolkit 使用外部 LLM API（例如 OpenAI GPT-4）。在代码中配置凭据：

```python
from evotoolkit.tools import HttpsApi

llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",  # 填写您的 API 地址
    key="your-api-key-here",  # 填写您的 API 密钥
    model="gpt-4o"
)
```

您可以从 OpenAI 获取 API Key，或使用其他兼容 OpenAI 协议的服务。

