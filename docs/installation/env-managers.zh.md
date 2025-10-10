# 环境与包管理器

使用您熟悉的工具管理 Python 环境。

---

## 使用 uv（推荐）

```bash
# 安装 uv
pip install uv

# 创建新项目
uv init my-evotool-project
cd my-evotool-project

# 添加 evotoolkit
uv add evotoolkit

# 运行脚本
uv run python main.py
```

---

## 使用 pip

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装 evotoolkit
pip install evotoolkit
```

---

## 使用 conda

```bash
# 创建 conda 环境
conda create -n evotool python=3.11
conda activate evotool

# 安装 evotoolkit
pip install evotoolkit
```

