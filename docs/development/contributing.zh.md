# 贡献指南

感谢你帮助改进 EvoToolkit。

---

## 可以贡献什么

- 报告 bug 或行为不一致
- 补充文档、教程和示例
- 为未覆盖行为补测试
- 提交聚焦明确的修复或新任务集成

---

## 开发环境

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/evotoolkit.git
cd evotoolkit
```

### 2. 安装开发依赖

```bash
uv sync --group dev

# 按需安装任务依赖
uv sync --extra scientific_regression
uv sync --extra prompt_engineering
uv sync --extra adversarial_attack
uv sync --extra cuda_engineering
uv sync --extra control_box2d
uv sync --extra cann_init
uv sync --extra all_tasks
```

### 3. 创建分支

```bash
git checkout -b feature/your-change
```

---

## 代码风格

项目当前使用 Ruff 做 lint 和格式化。

```bash
uv run ruff format .
uv run ruff check .
```

如果改动了公开行为，请同步补文档和测试。

---

## 提交前验证

在发起 PR 前，至少运行可移植测试子集：

```bash
uv run pytest tests/ -m "not cuda and not llm and not slow"
uv run mkdocs build
```

如果修改了打包或发布元数据，建议再验证：

```bash
uv build --out-dir dist
uvx twine check dist/*
```

---

## Pull Request 建议

1. 每个 PR 尽量只解决一件事。
2. 在描述里说明用户可见的变化或修复的问题。
3. 有对应 issue 或 discussion 时请关联。
4. 修改公开接口时同步更新文档和示例。

---

## 获取帮助

- GitHub Issues: <https://github.com/pgg3/evotoolkit/issues>
- GitHub Discussions: <https://github.com/pgg3/evotoolkit/discussions>
- Email: `pguo6680@gmail.com`
