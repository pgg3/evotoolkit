# 工具 API

EvoToolkit 的实用工具和辅助功能，包括 LLM API 客户端。

---

## HttpsApi

```python
class HttpsApi:
    def __init__(
        self,
        api_url: str,
        key: str,
        model: str,
        timeout: int = 300,
        temperature: float = 1.0
    )

    def get_response(self, messages: list) -> tuple[str, dict]
```

`HttpsApi` 类提供了一个简单的 HTTP(S) 客户端，用于与 LLM API（OpenAI、Claude 或兼容服务）交互。

**用途：**

- 连接到 LLM 聊天 API
- 发送提示并接收响应
- 自动处理身份验证和重试

**使用示例：**

```python
from evotoolkit.tools import HttpsApi
import os

# 创建 LLM API 客户端
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o",
    temperature=1.0
)

# 发送提示
messages = [{"role": "user", "content": "写一个计算斐波那契数列的 Python 函数"}]
response, usage = llm_api.get_response(messages)

print(response)
print(f"使用的 token 数: {usage['total_tokens']}")
```

**与 evotoolkit.solve() 一起使用：**

```python
import evotoolkit
from evotoolkit.task.python_task import ScientificRegressionTask, EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

# 配置 LLM
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o"
)

# 创建任务和接口
task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)

# 使用 LLM 求解
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

---

### 构造函数

```python
HttpsApi(api_url, key, model, embed_url=None, timeout=60, **kwargs)
```

**参数：**

- **api_url** (`str`): API 端点 URL，支持多种格式：
  - 完整 URL: `"https://api.openai.com/v1/chat/completions"`
  - 仅主机名: `"api.openai.com"` (默认为 `/v1/chat/completions`)

- **key** (`str`): 用于身份验证的 API 密钥
  - OpenAI: `"sk-..."`
  - Anthropic Claude: 您的 API 密钥
  - 自定义提供商: 查看提供商文档

- **model** (`str`): 要使用的模型名称
  - OpenAI: `"gpt-4o"`, `"gpt-4o-mini"`, `"gpt-3.5-turbo"`
  - Anthropic: `"claude-3-5-sonnet-20241022"`, `"claude-3-opus-20240229"`
  - Deepseek: `"deepseek-chat"`
  - 或任何兼容模型

- **embed_url** (`str`, 可选): 嵌入 API URL
  - 完整 URL: `"https://api.openai.com/v1/embeddings"`
  - 仅路径: `"/v1/embeddings"`
  - 如果未提供则自动推断

- **timeout** (`int`, 默认=60): 请求超时时间（秒）

- ****kwargs**: 附加参数
  - `temperature` (`float`, 默认=1.0): 采样温度 (0.0 到 2.0)
  - 其他提供商特定参数

**示例：**

```python
# OpenAI
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="sk-...",
    model="gpt-4o"
)

# Anthropic Claude (通过代理)
llm_api = HttpsApi(
    api_url="https://your-proxy.com/v1/chat/completions",
    key="sk-ant-...",
    model="claude-3-5-sonnet-20241022"
)

# 自定义提供商 (仅主机名)
llm_api = HttpsApi(
    api_url="api.custom-provider.com",
    key="your-key",
    model="custom-model"
)

# 自定义温度
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="sk-...",
    model="gpt-4o",
    temperature=0.7
)
```

---

### 方法

#### get_response()

```python
get_response(prompt: str | list[dict], *args, **kwargs) -> Tuple[str, dict]
```

向 LLM 发送提示并获取响应。

**参数：**

- **prompt** (`str` 或 `list[dict]`): 要发送的提示
  - 字符串: 简单文本提示
  - 字典列表: OpenAI 格式的聊天消息

**返回：**

- `tuple[str, dict]`: 一个元组 `(response_text, usage_info)`
  - `response_text` (`str`): 模型的响应
  - `usage_info` (`dict`): token 使用统计

**用法：**

```python
# 简单字符串提示
response, usage = llm_api.get_response("写一个 hello world 函数")

# 聊天消息
messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "解释 Python 装饰器"}
]
response, usage = llm_api.get_response(messages)

print(response)
print(f"Tokens: {usage['total_tokens']}")
```

**错误处理：**

- 失败时自动重试（最多 10 次）
- 超过最大重试次数后抛出 `RuntimeError`

---

#### get_embedding()

```python
get_embedding(text: str, *args, **kwargs) -> list[float]
```

获取文本字符串的嵌入向量。

**参数：**

- **text** (`str`): 要嵌入的输入文本

**返回：**

- `list[float]`: 嵌入向量

**用法：**

```python
embedding = llm_api.get_embedding("Hello world")
print(f"嵌入维度: {len(embedding)}")
```

**注意：** 需要配置 `embed_url`（对于常见提供商会自动推断）。

---

## 环境变量

### OPENAI_API_KEY

存储您的 OpenAI API 密钥：

```bash
export OPENAI_API_KEY="sk-..."
```

然后在 Python 中使用：

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

对于自定义配置：

```bash
export LLM_API_URL="https://your-api.com/v1/chat/completions"
export LLM_API_KEY="your-key"
export LLM_MODEL="gpt-4o"
```

---

## 最佳实践

### 应该做的 ✅

- 将 API 密钥存储在环境变量中（永远不要硬编码）
- 为您的用例使用适当的超时值
- 优雅地处理速率限制
- 监控 token 使用以控制成本
- 使用较低温度 (0.0-0.5) 获得确定性输出
- 使用较高温度 (0.7-1.5) 获得创造性输出

### 不应该做的 ❌

- 不要将 API 密钥提交到 git
- 不要使用过低的超时（< 30秒）
- 不要忽略 token 使用指标
- 不要在生产代码中禁用重试

---

## 提供商特定说明

### OpenAI

```python
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o"
)
```

- **模型**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **速率限制**: 查看您的账户等级
- **文档**: https://platform.openai.com/docs/api-reference

### Anthropic Claude

需要兼容的代理或 API 网关：

```python
llm_api = HttpsApi(
    api_url="https://your-gateway.com/v1/chat/completions",
    key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-3-5-sonnet-20241022"
)
```

- **模型**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- **注意**: 需要 OpenAI 兼容的 API 格式
- **文档**: https://docs.anthropic.com/

### 自定义提供商

许多 LLM 提供商提供 OpenAI 兼容的 API：

```python
llm_api = HttpsApi(
    api_url="api.custom-provider.com",  # 仅主机名
    key="your-key",
    model="provider-model-name"
)
```

查看提供商文档了解：
- API 端点 URL
- 身份验证方法
- 支持的模型
- 请求/响应格式

---

## 故障排除

### 连接错误

**问题:** `RuntimeError: Model Response Error!`

**解决方案:**
- 检查 API URL 是否正确
- 验证 API 密钥是否有效
- 确保网络连接
- 检查提供商状态页面

### 超时错误

**问题:** 请求超时

**解决方案:**
- 增加 `timeout` 参数
- 检查网络延迟
- 尝试较小的模型
- 降低提示复杂度

### 速率限制

**问题:** 请求过多

**解决方案:**
- 在请求之间添加延迟
- 减少并行度
- 升级 API 等级
- 实现指数退避

---

## 下一步

- 查看 [核心 API](core.md) 了解如何在 `evotoolkit.solve()` 中使用 LLM
- 查看 [方法 API](methods.md) 了解进化算法
- 尝试 [科学符号回归教程](../tutorials/built-in/scientific-regression.md)

