# 归档文档入口

这个 `docs/` 目录已经不再作为当前版本的主文档源，现仅保留为 `3.0.0` 之前文档体系的归档材料。

当前生效的文档站点改为从 `docs_core/` 构建，内容与新的核心运行时一致，重点包括：

- 显式创建算法对象并调用 `run()`
- `Method` 与 `MethodState`
- `RunStore`
- `checkpoint/state.pkl` 的 pickle checkpoint
- 具体 task 通过独立扩展包提供，而不是内置在 core 中

建议直接从下面这些页面开始看：

- `docs_core/index.md`
- `docs_core/quickstart.md`
- `docs_core/migration.md`
