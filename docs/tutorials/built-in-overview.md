# Built-in Tasks

EvoToolkit provides several pre-built optimization tasks that demonstrate the power of LLM-driven evolution across different domains.

## Available Tasks

### Scientific Regression
**[→ Start Tutorial](built-in/scientific-regression.md)**

Learn how to discover mathematical equations from real scientific datasets.

**You'll Learn:**
- Loading and working with scientific datasets
- Creating scientific regression tasks
- Using the high-level `evotoolkit.solve()` API
- Comparing different evolutionary algorithms (EoH, EvoEngineer, FunSearch)
- Interpreting discovered equations

**Prerequisites:** Basic Python and NumPy knowledge

---

### Prompt Engineering
**[→ Start Tutorial](built-in/prompt-engineering.md)**

Optimize LLM prompts to improve task performance.

**You'll Learn:**
- LLM prompt optimization basics
- Using string optimization tasks
- Evolving prompt templates
- Evaluating and comparing different prompts

**Prerequisites:** Scientific Regression tutorial

---

### Adversarial Attack
**[→ Start Tutorial](built-in/adversarial-attack.md)**

Learn how to evolve adversarial examples and attack algorithms.

**You'll Learn:**
- Creating adversarial attack tasks
- Evolving attack strategies
- Generating adversarial examples
- Evaluating attack effectiveness

**Prerequisites:** Scientific Regression tutorial, machine learning basics

---

### CUDA Tasks
**[→ Start Tutorial](built-in/cuda-task.md)**

Optimize GPU kernels using LLM-driven evolution.

**You'll Learn:**
- Creating CUDA optimization tasks
- Benchmarking GPU performance
- Evolving efficient CUDA kernels
- Handling compilation and execution

**Prerequisites:** CUDA programming basics, GPU hardware

---

### Control Box2D (Lunar Lander)
**[→ Start Tutorial](built-in/control-box2d.md)**

Evolve interpretable control policies for the Gymnasium LunarLander-v3 environment.

**You'll Learn:**
- Creating control tasks for physical simulation
- Evolving human-readable Python policies
- Understanding the `policy(state) -> action` interface
- Evaluating policies across multiple episodes

**Prerequisites:** Basic Python knowledge

---

### CANN Init (Ascend NPU)
**[→ Start Tutorial](built-in/cann-init.md)**

Generate Ascend C operator kernel code for Huawei Ascend NPUs.

**You'll Learn:**
- Creating CANN Init tasks with Python references
- Evolving Ascend C kernel implementations
- Understanding the template system
- Compiling and evaluating on Ascend hardware

**Prerequisites:** Huawei CANN toolkit, Ascend NPU hardware

---

## Task Comparison

| Task | Domain | Difficulty | Best For |
|------|--------|------------|----------|
| Scientific Regression | Data Science | Beginner | Learning the basics, equation discovery |
| Prompt Engineering | NLP/LLM | Intermediate | Optimizing LLM interactions |
| Adversarial Attack | Security/ML | Intermediate | Security research, robustness testing |
| CUDA Tasks | GPU Computing | Advanced | Performance optimization |
| Control Box2D | Robotics/Control | Intermediate | Interpretable control policy discovery |
| CANN Init | Ascend NPU | Advanced | Ascend C operator generation (requires hardware) |

---

## Getting Started

1. **Start with Scientific Regression** if you're new to EvoToolkit
2. **Try Prompt Engineering** to see how evolution can optimize text
3. **Explore Adversarial Attack** for security applications
4. **Master CUDA Tasks** for GPU optimization

Each tutorial includes complete, runnable code examples that you can adapt for your own problems.