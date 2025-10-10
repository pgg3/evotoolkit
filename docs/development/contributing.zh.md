# 贡献指南

感谢您对 EvoToolkit 的贡献兴趣！本指南将帮助您开始。

---

## 开发设置

### 1. Fork 和克隆仓库

```bash
# Fork 仓库到您的 GitHub 账户
# 然后克隆您的 fork
git clone https://github.com/YOUR_USERNAME/evotoolkit.git
cd evotool
```

### 2. 安装开发依赖

```bash
# 安装开发依赖
uv sync --group dev

# 这会安装:
# - black (代码格式化)
# - isort (导入排序)
# - mypy (类型检查)
# - mkdocs (文档)

# 如果需要开发特定任务，可以安装可选依赖:
uv sync --extra cuda_engineering       # CUDA 任务
uv sync --extra scientific_regression  # 科学符号回归
uv sync --extra adversarial_attack     # 对抗攻击
uv sync --extra all_tasks              # 所有任务依赖
```

### 3. 创建分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

---

## 开发工作流

### 代码格式化

```bash
# 格式化代码
uv run black .

# 排序导入
uv run isort .

# 类型检查
uv run mypy src/evotool
```

### 构建文档

```bash
# 在本地服务文档
uv run mkdocs serve

# 在浏览器中打开 http://127.0.0.1:8000
```

---

## 贡献类型

### Bug 修复

1. 在 [GitHub Issues](https://github.com/pgg3/evotoolkit/issues) 中创建 issue
2. 描述 bug 和重现步骤
3. 提交修复的 pull request

### 新功能

1. 首先在 [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions) 中讨论
2. 获得维护者批准后创建 issue
3. 实现功能并添加测试
4. 更新文档
5. 提交 pull request

### 文档改进

1. 识别需要改进的文档
2. 在 `docs/` 中进行更改
3. 本地测试（`mkdocs serve`）
4. 提交 pull request

---

## Pull Request 流程

### 1. 准备您的更改

```bash
# 确保测试通过
uv run pytest

# 格式化代码
uv run black .
uv run isort .

# 检查类型
uv run mypy src/evotool
```

### 2. 提交更改

```bash
git add .
git commit -m "简短描述性的提交信息"

# 提交信息格式:
# feat: 添加新功能
# fix: 修复 bug
# docs: 文档更改
# test: 测试更改
# refactor: 代码重构
```

### 3. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 pull request。

### 4. PR 检查清单

- [ ] 所有测试通过
- [ ] 代码已格式化（black、isort）
- [ ] 类型检查通过（mypy）
- [ ] 添加了新功能的文档
- [ ] 添加了新代码的测试
- [ ] PR 描述清晰说明更改

---

## 代码风格

### Python 代码

- 遵循 [PEP 8](https://pep8.org/)
- 使用 Black 进行格式化（行长度：88）
- 为所有公共 API 添加类型提示
- 为函数和类编写文档字符串

示例：

```python
def evaluate_solution(solution: Solution, task: BaseTask) -> float:
    """评估给定任务的解。

    Args:
        solution: 要评估的候选解
        task: 定义评估标准的任务

    Returns:
        适应度值（越低越好）

    Raises:
        ValueError: 如果解无效
    """
    # 实现
    pass
```

### 测试

- 为所有新功能编写测试
- 使用描述性测试名称
- 遵循 Arrange-Act-Assert 模式

示例：

```python
def test_scientific_regression_task_evaluation():
    """测试 ScientificRegressionTask 是否正确评估有效方程。"""
    # Arrange
    task = ScientificRegressionTask(dataset_name="bactgrow")
    code = '''import numpy as np
    def equation(b, s, temp, pH, params):
        return params[0] * b + params[1]
    '''

    # Act
    result = task.evaluate_code(code)

    # Assert
    assert result.valid
    assert result.score > 0
```

---

## 文档

### 文档字符串

使用 Google 风格的文档字符串：

```python
def my_function(param1: str, param2: int) -> bool:
    """单行摘要。

    更详细的描述（可选）。

    Args:
        param1: 第一个参数的描述
        param2: 第二个参数的描述

    Returns:
        返回值的描述

    Raises:
        ValueError: 何时引发此异常
    """
    pass
```

### Markdown 文档

- 使用清晰的标题层次结构
- 包含代码示例
- 添加到 `mkdocs.yml` 导航

---

## 发布流程

（仅维护者）

1. 更新 `pyproject.toml` 中的版本
2. 更新 `CHANGELOG.md`
3. 创建 git 标签
4. 推送到 PyPI

---

## 获取帮助

- **问题**: [GitHub Issues](https://github.com/pgg3/evotoolkit/issues)
- **讨论**: [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions)
- **电子邮件**: pguo6680@gmail.com

---

## 行为准则

请友善和尊重。我们希望为每个人营造一个欢迎的环境。

---

## 许可证

通过贡献，您同意您的贡献将根据与项目相同的许可证进行许可（参见 [LICENSE](https://github.com/pgg3/evotoolkit/blob/master/LICENSE)）。
