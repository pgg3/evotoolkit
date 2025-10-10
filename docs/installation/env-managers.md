# Environments & Package Managers

Manage your Python environment with your preferred tool.

---

## Using uv (Recommended)

```bash
# Install uv
pip install uv

# Create a new project
uv init my-evotool-project
cd my-evotool-project

# Add evotoolkit
uv add evotoolkit

# Run your script
uv run python main.py
```

---

## Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install evotoolkit
pip install evotoolkit
```

---

## Using conda

```bash
# Create conda environment
conda create -n evotool python=3.11
conda activate evotool

# Install evotoolkit
pip install evotoolkit
```

