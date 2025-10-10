# Adversarial Attack Tutorial

Learn how to use LLM-driven evolution to discover effective adversarial attack algorithms.

!!! note "Academic Citation"
    The adversarial attack task is based on L-AutoDA research. If you use this feature in academic work, please cite:

    ```bibtex
    @inproceedings{10.1145/3638530.3664121,
        author = {Guo, Ping and Liu, Fei and Lin, Xi and Zhao, Qingchuan and Zhang, Qingfu},
        title = {L-AutoDA: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks},
        year = {2024},
        isbn = {9798400704956},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3638530.3664121},
        doi = {10.1145/3638530.3664121},
        pages = {1846–1854},
        numpages = {9},
        keywords = {large language models, adversarial attacks, automated algorithm design, evolutionary algorithms},
        location = {Melbourne, VIC, Australia},
        series = {GECCO '24 Companion}
    }
    ```

!!! tip "Complete Example Code"
    This tutorial provides complete, runnable examples (click to view/download):

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/adversarial_attack/basic_example.py) - Basic usage example
    - [:material-file-document: README.md](https://github.com/pgg3/evotoolkit/blob/master/examples/adversarial_attack/README.md) - Examples documentation and usage guide

    Run locally:
    ```bash
    cd examples/adversarial_attack
    python basic_example.py
    ```

---

## Overview

This tutorial demonstrates:

- Creating adversarial attack tasks
- Using LLM-driven evolution to discover attack algorithms
- Understanding the `draw_proposals` function
- Evaluating attacks on neural networks
- Evolving effective black-box attacks automatically

---

## Installation

!!! tip "GPU Recommended"
    For best performance, install PyTorch with CUDA support before EvoToolkit.
    We recommend **CUDA 12.9** (latest stable).

### Step 1: Install PyTorch with GPU Support

```bash
# CUDA 12.9 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# For other versions, visit: https://pytorch.org/get-started/locally/
# CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended, slower performance)
# pip install torch torchvision
```

### Step 2: Install EvoToolkit

```bash
pip install evotoolkit[adversarial_attack]
```

This installs:

- `timm` - PyTorch Image Models (provides CIFAR-10 pretrained models from Hugging Face)
- `foolbox` - Adversarial attacks library

**Prerequisites:**

- Python >= 3.11
- PyTorch >= 2.0 (with CUDA support recommended)
- LLM API access (OpenAI, Claude, or other compatible providers)
- Basic understanding of adversarial machine learning

---

## Understanding Adversarial Attack Tasks

### What is an Adversarial Attack Task?

An adversarial attack task evolves **proposal generation algorithms** to create adversarial examples that fool neural networks with minimal distortion.

| Aspect | Scientific Regression | Adversarial Attack |
|--------|----------------------|---------------------|
| **Solution type** | Mathematical equation | Proposal algorithm |
| **Function name** | `equation` | `draw_proposals` |
| **Inputs** | Data + params | Images + noise + hyperparams |
| **Evaluation** | MSE on predictions | L2 distance of adversarials |
| **Goal** | Minimize prediction error | Minimize distortion |

### Task Components

An adversarial attack task requires:

- **Target model**: Neural network to attack
- **Test data**: Images to generate adversarial examples for
- **Attack budget**: Number of iterations/queries
- **Evaluation metric**: L2 distance between original and adversarial images

---

## Creating Your First Adversarial Attack Task

### Step 1: Load Target Model and Data

```python
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms

# Load CIFAR-10 pretrained ResNet18 model (from Hugging Face Hub)
# This model achieves 94.98% accuracy on CIFAR-10
# CIFAR-10 ResNet18 uses modified architecture (3x3 conv1, removed maxpool)
model = timm.create_model("resnet18", num_classes=10, pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# Load pretrained weights
model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet18_cifar10.pth"
    )
)
model.eval()

if torch.cuda.is_available():
    model.cuda()

# Load CIFAR-10 test set
# Use CIFAR-10 standard normalization parameters
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2471, 0.2435, 0.2616])
])
test_set = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)
```

### Step 2: Create Task and Test Initial Solution

```python
from evotoolkit.task.python_task import AdversarialAttackTask

# Create task
task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=1000,
    n_test_samples=10,
    use_mock=False
)

# Get initial solution
init_sol = task.make_init_sol_wo_other_info()

print(f"Initial algorithm:")
print(init_sol.sol_string)
print(f"\nScore: {init_sol.evaluation_res.score:.2f}")
print(f"Avg L2 distance: {init_sol.evaluation_res.additional_info['avg_distance']:.2f}")
```

**Output:**
```
Initial algorithm:
import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """Baseline proposal generation..."""
    ...

Score: -2.34
Avg L2 distance: 2.34
```

### Step 3: Test Custom Algorithm

```python
custom_code = '''import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """Simple algorithm: move toward original with noise."""
    org = org_img.flatten()
    best = best_adv_img.flatten()
    noise = std_normal_noise.flatten()

    # Move toward original with random perturbation
    direction = org - best
    step = hyperparams[0] * 0.1
    candidate = best + step * direction + step * noise * 0.5

    return candidate.reshape(org_img.shape)
'''

result = task.evaluate_code(custom_code)
print(f"Score: {result.score:.2f}")
print(f"Avg L2 distance: {result.additional_info['avg_distance']:.2f}")
```

---

## Understanding the draw_proposals Function

### Function Signature

The evolved function must have this exact signature:

```python
def draw_proposals(
    org_img: np.ndarray,         # Original clean image
    best_adv_img: np.ndarray,    # Current best adversarial
    std_normal_noise: np.ndarray,# Random noise for exploration
    hyperparams: np.ndarray      # Adaptive step size
) -> np.ndarray:                 # New candidate adversarial
    """Generate new candidate adversarial example."""
    ...
```

### Input Details

**org_img** (Original Image):
- Shape: `(3, H, W)` for RGB images (e.g., `(3, 32, 32)` for CIFAR-10)
- Values: `[0, 1]` normalized pixel values
- Purpose: The clean image we're attacking

**best_adv_img** (Best Adversarial):
- Shape: `(3, H, W)` - same as org_img
- Values: `[0, 1]`
- Purpose: Current best adversarial example (fools the model, closest to original)

**std_normal_noise** (Random Noise):
- Shape: `(3, H, W)` - same as org_img
- Values: Sampled from standard normal distribution N(0, 1)
- Purpose: Provides randomness for exploration

**hyperparams** (Adaptive Parameters):
- Shape: `(1,)` - single scalar value
- Values: Typically in range `[0.5, 1.5]`
- Purpose: Adaptive step size that increases when finding adversarials

### Return Value

Must return a numpy array with:
- Shape: `(3, H, W)` - same as org_img
- Values: Any (will be clipped to `[0, 1]` automatically)
- Purpose: New candidate adversarial example

### Algorithm Design Principles

**1. Exploitation (Refinement)**

Move along the direction from org_img toward decision boundary:

```python
direction = org_img - best_adv_img
candidate = best_adv_img + step_size * direction
```

**2. Exploration (Discovery)**

Add random noise to discover new regions:

```python
candidate = best_adv_img + noise_component
```

**3. Adaptive Step Size**

Use hyperparams to balance exploration/exploitation:

```python
# hyperparams increases when finding adversarials
step = hyperparams[0] * base_step_size
```

**4. Complete Example**

```python
import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """Combine parallel and perpendicular components."""
    # Flatten to vectors
    org = org_img.flatten()
    best = best_adv_img.flatten()
    noise = std_normal_noise.flatten()

    # Compute direction
    direction = org - best
    direction_norm = np.linalg.norm(direction)

    # Parallel component (toward original)
    noise_norm = np.linalg.norm(noise)
    step_size = (noise_norm * hyperparams[0]) ** 2
    d_parallel = step_size * direction

    # Perpendicular component (exploration)
    if direction_norm > 1e-8:
        dot_product = np.dot(direction, noise)
        projection = (dot_product / direction_norm) * direction
        d_perpendicular = (projection / direction_norm - direction_norm * noise) * hyperparams[0]
    else:
        d_perpendicular = noise * hyperparams[0]

    # Combine
    candidate = best + d_parallel + d_perpendicular

    return candidate.reshape(org_img.shape)
```

---

## Running Evolution to Discover Attacks

### Step 1: Create Interface

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi

# Create interface
interface = EvoEngineerPythonInterface(task)
```

### Step 2: Configure LLM

```python
llm_api = HttpsApi(
    api_url="api.openai.com",  # Your API URL
    key="your-api-key-here",   # Your API key
    model="gpt-4o"
)
```

### Step 3: Run Evolution

```python
result = evotoolkit.solve(
    interface=interface,
    output_path='./attack_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

print(f"Best algorithm found:")
print(result.sol_string)
print(f"\nAvg L2 distance: {-result.evaluation_res.score:.2f}")
```

!!! tip "Try Different Algorithms"
    EvoToolkit supports multiple evolutionary algorithms for adversarial attacks:

    ```python
    # Using EoH
    from evotoolkit.task.python_task import EoHPythonInterface
    interface = EoHPythonInterface(task)

    # Using FunSearch
    from evotoolkit.task.python_task import FunSearchPythonInterface
    interface = FunSearchPythonInterface(task)

    # Using EvoEngineer (default)
    from evotoolkit.task.python_task import EvoEngineerPythonInterface
    interface = EvoEngineerPythonInterface(task)
    ```

    Then use the same `evotoolkit.solve()` call to run evolution. Different interfaces may discover different attack strategies.

---

## Attack Evolution Example

During evolution, the LLM discovers increasingly effective algorithms:

**Generation 1: Simple baseline**
```python
def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    return best_adv_img + 0.01 * std_normal_noise
# Avg L2: 3.5
```

**Generation 3: Direction-based**
```python
def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    direction = org_img - best_adv_img
    return best_adv_img + hyperparams[0] * 0.1 * direction
# Avg L2: 2.1
```

**Generation 7: Sophisticated combination**
```python
def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    # Complex algorithm combining multiple components
    ...
# Avg L2: 0.8
```

---

## Customizing Evolution Behavior

The quality of evolved attacks is controlled by the **evolution method** and its internal **prompt design**. To improve results:

- **Adjust prompts**: Inherit existing Interface classes and customize LLM prompts
- **Develop new algorithms**: Create entirely new evolutionary strategies

!!! tip "Learn More"
    These are general techniques applicable to all tasks. For detailed tutorials, see:

    - **[Customizing Evolution Methods](../customization/customizing-evolution.md)** - How to modify prompts and develop new algorithms
    - **[Advanced Usage](../advanced-overview.md)** - More advanced configuration options

**Quick Example - Custom Prompts for Adversarial Attacks:**

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface

class CustomAttackInterface(EvoEngineerPythonInterface):
    """Interface optimized for adversarial attack evolution."""

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """Customize mutation prompt to emphasize attack effectiveness."""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            individual = selected_individuals[0]

            prompt = f"""# Adversarial Attack Algorithm Evolution

{task_description}

## Current Best Algorithm
**Avg L2 Distance:** {-current_best_sol.evaluation_res.score:.2f}
**Algorithm:** {current_best_sol.sol_string}

## Algorithm to Mutate
**Avg L2 Distance:** {-individual.evaluation_res.score:.2f}
**Algorithm:** {individual.sol_string}

## Optimization Guidelines
Focus on improving the algorithm by:
- Better balancing exploitation (refinement) and exploration (discovery)
- More effective use of the adaptive hyperparams
- Clever combination of direction vectors and noise
- Numerical stability and efficiency

Generate an improved draw_proposals function that achieves lower L2 distances.

## Response Format:
name: [descriptive_name]
code:
[Your improved draw_proposals function]
thought: [reasoning for changes]
"""
            return [{"role": "user", "content": prompt}]

        # Use default for other operators
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# Use custom interface
interface = CustomAttackInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./custom_results',
    running_llm=llm_api,
    max_generations=10
)
```

---

## Understanding Evaluation

### Scoring Mechanism

1. **Attack Execution**: Run evolved algorithm on test samples
2. **Adversarial Generation**: Create adversarial examples using draw_proposals
3. **Distance Measurement**: Compute L2 distance from original images
4. **Fitness Calculation**: Score = -(average L2 distance)

Lower L2 distance = better attack = higher score (less negative)

### Evaluation Output

```python
result = task.evaluate_code(algorithm_code)

if result.valid:
    print(f"Score: {result.score:.2f}")  # Negative L2 distance
    print(f"Avg L2: {result.additional_info['avg_distance']:.2f}")
    print(f"Attack steps: {result.additional_info['attack_steps']}")
else:
    print(f"Error: {result.additional_info['error']}")
```

---

## Use Cases and Applications

### 1. Black-Box Attack Discovery

Evolve algorithms for black-box scenarios where gradients are unavailable:

```python
task = AdversarialAttackTask(
    model=black_box_model,
    test_loader=test_loader,
    attack_steps=5000,  # More iterations for black-box
    n_test_samples=50
)
```

### 2. Robustness Evaluation

Test model defenses by evolving strong attacks:

```python
# Load more robust model (e.g., adversarially trained)
# Note: You need to train or obtain robust models yourself
from torchvision import models
model = models.resnet50(pretrained=True)  # Or your robust model
model.eval()

task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=10000,  # Thorough evaluation
    n_test_samples=100
)
```

!!! note "About Robust Models"
    EvoToolkit no longer depends on robustbench library. If you need to test robust models:

    - Use your own adversarially trained models
    - Load pretrained robust models from other sources
    - Or use standard models for basic testing

### 3. Transfer Attack Development

Evolve attacks that transfer across models:

```python
# Train on surrogate model
from torchvision import models
surrogate_model = models.resnet18(pretrained=True)
surrogate_model.eval()

task = AdversarialAttackTask(
    model=surrogate_model,
    test_loader=test_loader,
    attack_steps=5000,
    n_test_samples=50
)

# Evolve attack
result = evotoolkit.solve(interface, ...)

# Test on target model
target_model = models.resnet50(pretrained=True)  # Different architecture
target_model.eval()
# Evaluate evolved algorithm on target_model
```

### 4. Query-Efficient Attacks

Optimize for minimal queries to the target model:

```python
task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=100,  # Limited queries
    n_test_samples=20
)
```

---

## Complete Example

Here's a full working example:

```python
import torch
import torch.nn as nn
import timm
import evotoolkit
from torchvision import datasets, transforms
from evotoolkit.task.python_task import (
    AdversarialAttackTask,
    EvoEngineerPythonInterface
)
from evotoolkit.tools.llm import HttpsApi

# 1. Load CIFAR-10 pretrained ResNet18 model
model = timm.create_model("resnet18", num_classes=10, pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# Load pretrained weights
model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet18_cifar10.pth"
    )
)
model.eval()

if torch.cuda.is_available():
    model.cuda()

# 2. Prepare test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2471, 0.2435, 0.2616])
])
test_set = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=32, shuffle=False
)

# 3. Create task
task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=1000,
    n_test_samples=10,
    use_mock=False
)

# 4. Create LLM API
llm_api = HttpsApi(
    api_url="api.openai.com",  # Your API URL
    key="your-api-key-here",   # Your API key
    model="gpt-4o"
)

# 5. Create interface
interface = EvoEngineerPythonInterface(task)

# 6. Run evolution
result = evotoolkit.solve(
    interface=interface,
    output_path='./attack_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

# 7. Show results
print(f"Best attack algorithm found:")
print(result.sol_string)
print(f"\nAvg L2 distance: {-result.evaluation_res.score:.2f}")
print(f"Attack steps: {result.evaluation_res.additional_info['attack_steps']}")
```

---

## Next Steps

### Explore Different Attack Scenarios

- Try different target models (standard vs robust)
- Experiment with different datasets (CIFAR-10, ImageNet)
- Compare different evolutionary algorithms
- Test evolved attacks on multiple models

### Customize and Improve Evolution

- Examine prompt designs in existing Interface classes
- Inherit and override Interfaces to customize prompts
- Design specialized prompts for different attack types
- Develop new evolutionary algorithms if needed

### Learn More

- [Customizing Evolution Methods](../customization/customizing-evolution.md) - Deep dive into prompt customization
- [Advanced Usage](../advanced-overview.md) - Advanced configuration and techniques
- [API Reference](../../api/index.md) - Complete API documentation
- [L-AutoDA Paper](https://doi.org/10.1145/3638530.3664121) - GECCO 2024

---

## References

- **L-AutoDA**: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks (GECCO 2024)
- **Foolbox**: A Python toolbox to create adversarial examples
- **PyTorch Models**: Pretrained computer vision models (https://pytorch.org/vision/stable/models.html)
