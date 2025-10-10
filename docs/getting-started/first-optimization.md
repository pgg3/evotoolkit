# Your First Optimization: Scientific Symbolic Regression

Let's use EvoToolkit to discover mathematical equations from real scientific data.

!!! note "Academic Citation"
    The scientific regression task and datasets used in this guide are based on research from CoEvo. If you use this feature in academic work, please cite:

    ```bibtex
    @misc{guo2024coevocontinualevolutionsymbolic,
        title={CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models},
        author={Ping Guo and Qingfu Zhang and Xi Lin},
        year={2024},
        eprint={2412.18890},
        archivePrefix={arXiv},
        primaryClass={cs.AI},
        url={https://arxiv.org/abs/2412.18890}
    }
    ```

---

## Step 1: Create a New Project

```bash
mkdir my-evotool-project
cd my-evotool-project
```

## Step 2: Write Your First Script

Create a file named `first_optimization.py`:

```python
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

# Step 1: Create a task
print("Creating scientific regression task...")
task = ScientificRegressionTask(dataset_name="bactgrow")

# Step 2: Create an interface
print("Setting up EvoEngineer interface...")
interface = EvoEngineerPythonInterface(task)

# Step 3: Configure LLM API
print("Configuring LLM API...")
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",  # Your API endpoint
    key="your-api-key-here",  # Your API key
    model="gpt-4o"
)

# Step 4: Run optimization
print("\nStarting optimization with EvoEngineer...")
print("This may take a few minutes...\n")

result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    max_sample_nums=10,
    pop_size=5
)

print("\n" + "="*60)
print("Optimization completed!")
print("="*60)
print(f"\nBest solution fitness: {result.evaluation_res.score}")
print(f"Results saved to: ./results/")
```

## Step 3: Run the Script

```bash
python first_optimization.py
```

You should see output similar to:

```
Creating scientific regression task...
Setting up EvoEngineer interface...
Configuring LLM API...

Starting optimization with EvoEngineer...
This may take a few minutes...

Generation 1/5: Best fitness = 0.245
Generation 2/5: Best fitness = 0.189
Generation 3/5: Best fitness = 0.134
Generation 4/5: Best fitness = 0.098
Generation 5/5: Best fitness = 0.067

============================================================
Optimization completed!
============================================================

Best solution fitness: 0.067
Results saved to: ./results/
```

---

Next: [Understanding the Code](understanding-code.md)
