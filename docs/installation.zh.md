# 安装

需要 Python 3.10 或更新版本。

```bash
pip install evotoolkit
```

从源码仓库开发：

```bash
uv sync --group dev
uv run pytest tests/
```

核心仅依赖 NumPy。ChatProposer 可接收任意提供 get_response(messages) 的客户端；内置 HttpsApi 使用 chat-completions 格式。真实模型访问需要配置自己的地址和密钥。

AgentProposer 另需安装 headless-cli 并认证对应后端，详见[扩展](extensions.md)。导入 EvoToolkit 不会安装 agent 依赖。
