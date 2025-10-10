# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Scientific Symbolic Regression Task implementation.

This task discovers mathematical functions from real scientific datasets.
Adapted from LLM-SR (CoEvo project).
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Literal
from evotoolkit.core import Solution, EvaluationResult
from evotoolkit.task.python_task.python_task import PythonTask
from evotoolkit.registry import register_task


# Dataset metadata
DATASET_INFO = {
    "bactgrow": {
        "description": "E. Coli bacterial growth rate prediction",
        "inputs": ["b (population density)", "s (substrate concentration)",
                   "temp (temperature)", "pH (pH level)"],
        "output": "db (growth rate)",
        "input_cols": ["b", "s", "temp", "pH"],
        "output_col": "db"
    },
    "oscillator1": {
        "description": "Damped nonlinear oscillator acceleration",
        "inputs": ["x (position)", "v (velocity)"],
        "output": "dv (acceleration)",
        "input_cols": ["x", "v"],
        "output_col": "dv"
    },
    "oscillator2": {
        "description": "Damped nonlinear oscillator (variant 2)",
        "inputs": ["x (position)", "v (velocity)"],
        "output": "dv (acceleration)",
        "input_cols": ["x", "v"],
        "output_col": "dv"
    },
    "stressstrain": {
        "description": "Stress prediction in Aluminium rod",
        "inputs": ["strain", "temp (temperature)"],
        "output": "stress",
        "input_cols": ["strain", "temp"],
        "output_col": "stress"
    }
}


@register_task("ScientificRegression")
class ScientificRegressionTask(PythonTask):
    """
    Scientific symbolic regression task for discovering mathematical equations.

    This task evaluates Python code that defines an `equation` function,
    optimizes its parameters using scipy, and returns fitness based on MSE.
    """

    def __init__(
        self,
        dataset_name: Literal["bactgrow", "oscillator1", "oscillator2", "stressstrain"],
        data_dir: str | Path | None = None,
        max_params: int = 10,
        timeout_seconds: float = 60.0
    ):
        """
        Initialize scientific regression task.

        Args:
            dataset_name: Name of the scientific dataset
            data_dir: Custom data directory (optional, defaults to ~/.evotool/data/)
            max_params: Maximum number of optimizable parameters
            timeout_seconds: Execution timeout
        """
        if dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )

        self.dataset_name = dataset_name
        self.max_params = max_params
        self.dataset_info = DATASET_INFO[dataset_name]

        # Load data
        train_data, test_data = self._load_dataset(dataset_name, data_dir)

        # Store data
        self.train_inputs = train_data['inputs']
        self.train_outputs = train_data['outputs']
        self.test_inputs = test_data['inputs']
        self.test_outputs = test_data['outputs']

        # Pass to parent
        super().__init__(
            data={'train': train_data, 'test': test_data},
            timeout_seconds=timeout_seconds
        )

    def _load_dataset(self, dataset_name: str, data_dir: str | Path | None):
        """
        Load dataset, automatically downloading from GitHub release if needed.

        Returns:
            tuple: (train_data, test_data) dictionaries with 'inputs' and 'outputs'
        """
        from evotoolkit.data import get_dataset_path, DownloadError

        try:
            # Get dataset path, will auto-download if needed
            base_dir = get_dataset_path('scientific_regression', data_dir=data_dir)
            dataset_path = base_dir / dataset_name
        except DownloadError as e:
            raise FileNotFoundError(
                f"Failed to download dataset '{dataset_name}': {str(e)}"
            ) from e

        # Verify dataset exists
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found after download. "
                f"This might be a bug - please report it at: "
                f"https://github.com/pgg3/evotoolkit/issues"
            )

        # Load CSV files
        info = self.dataset_info
        train_df = pd.read_csv(dataset_path / "train.csv")
        test_df = pd.read_csv(dataset_path / "test_id.csv")  # Use in-distribution test

        # Extract inputs and outputs
        train_data = {
            'inputs': train_df[info['input_cols']].values,
            'outputs': train_df[info['output_col']].values
        }
        test_data = {
            'inputs': test_df[info['input_cols']].values,
            'outputs': test_df[info['output_col']].values
        }

        return train_data, test_data

    def _process_data(self, data):
        """Process input data and create task_info."""
        self.data = data
        self.task_info = {
            'dataset_name': self.dataset_name,
            'train_size': len(data['train']['inputs']),
            'test_size': len(data['test']['inputs']),
            'n_inputs': data['train']['inputs'].shape[1],
            'max_params': self.max_params
        }

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """
        Evaluate Python code for scientific symbolic regression.

        The code must define an `equation` function that will be optimized.
        """
        # Create namespace with required modules
        namespace = {
            '__builtins__': {
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'print': print, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                '__import__': __import__,
            },
            'np': np,
        }

        # Execute the code
        exec(candidate_code, namespace)

        # Check if equation function exists
        if 'equation' not in namespace:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Function "equation" not found in code'}
            )

        equation_func = namespace['equation']

        # Evaluate on training data
        try:
            train_score, train_warnings = self._evaluate_equation(
                equation_func,
                self.train_inputs,
                self.train_outputs
            )

            # Evaluate on test data
            test_score, test_warnings = self._evaluate_equation(
                equation_func,
                self.test_inputs,
                self.test_outputs
            )

            # Combine all warnings
            all_warnings = list(set(train_warnings + test_warnings))

            if train_score is None or test_score is None:
                return EvaluationResult(
                    valid=False,
                    score=float('-inf'),
                    additional_info={
                        'error': 'Optimization failed or returned NaN/Inf',
                        'warnings': all_warnings
                    }
                )

            # Use train_score as fitness (already -MSE, higher is better)
            # Test score is only for final evaluation, not for optimization
            score = train_score

            return EvaluationResult(
                valid=True,
                score=score,
                additional_info={
                    'train_mse': -train_score,  # Convert -MSE back to MSE for logging
                    'test_mse': -test_score,     # Convert -MSE back to MSE for logging
                    'n_params': self.max_params,
                    'warnings': all_warnings if all_warnings else []
                }
            )

        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': f'Evaluation error: {str(e)}'}
            )

    def _evaluate_equation(self, equation_func, inputs, outputs):
        """
        Evaluate equation with parameter optimization.

        Returns:
            tuple: (score, warnings_list) where score is -MSE (higher is better), or (None, warnings) if failed
        """
        from scipy.optimize import minimize

        captured_warnings = []

        # Define loss function
        def loss(params):
            try:
                # Call equation with unpacked inputs and params
                if inputs.shape[1] == 2:
                    y_pred = equation_func(inputs[:, 0], inputs[:, 1], params)
                elif inputs.shape[1] == 4:
                    y_pred = equation_func(inputs[:, 0], inputs[:, 1],
                                           inputs[:, 2], inputs[:, 3], params)
                else:
                    # Generic case
                    y_pred = equation_func(*[inputs[:, i] for i in range(inputs.shape[1])], params)

                mse = np.mean((y_pred - outputs) ** 2)
                return mse
            except:
                return 1e10  # Large penalty for errors

        # Optimize parameters with warning capture
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                result = minimize(
                    loss,
                    x0=[1.0] * self.max_params,
                    method='BFGS',
                    options={'maxiter': 1000}
                )

                # Collect warning messages
                for warning in w:
                    msg = f"{warning.category.__name__}: {warning.message}"
                    if msg not in captured_warnings:  # Deduplicate
                        captured_warnings.append(msg)

            final_loss = result.fun

            # Check for NaN or Inf
            if np.isnan(final_loss) or np.isinf(final_loss):
                return (None, captured_warnings)

            # Return negative MSE (higher is better) and warnings
            return (-final_loss, captured_warnings)

        except:
            return (None, captured_warnings)

    def get_base_task_description(self) -> str:
        """Get task description for the specific dataset."""
        info = self.dataset_info
        input_names = info['input_cols']
        output_name = info['output_col']

        # Build input signature
        if len(input_names) == 2:
            signature = f"{input_names[0]}: np.ndarray, {input_names[1]}: np.ndarray, params: np.ndarray"
        elif len(input_names) == 4:
            signature = ", ".join([f"{name}: np.ndarray" for name in input_names]) + ", params: np.ndarray"
        else:
            signature = ", ".join([f"input{i}: np.ndarray" for i in range(len(input_names))]) + ", params: np.ndarray"

        return f"""You are an expert in scientific symbolic regression and mathematical modeling.

Task: {info['description']}

Your goal is to discover a mathematical equation that predicts {output_name} from:
{chr(10).join(f"  - {inp}" for inp in info['inputs'])}

Requirements:
- Define a function named 'equation' with signature: equation({signature}) -> np.ndarray
- Use numpy operations for vectorized computation
- The 'params' array contains {self.max_params} optimizable constants (params[0] to params[{self.max_params-1}])
- Return predictions as a numpy array matching the shape of inputs
- Focus on discovering the mathematical structure; parameters will be auto-optimized

Guidelines:
- Use mathematical operations: +, -, *, /, **, np.exp, np.log, np.sin, np.cos, etc.
- Combine input variables in meaningful ways based on physical intuition
- Keep equations reasonably simple to avoid overfitting
- Ensure numerical stability (avoid division by very small numbers, etc.)
- All operations must be vectorized (work on numpy arrays)

Example structure:
```python
import numpy as np

def equation({signature}) -> np.ndarray:
    # Example: linear combination
    return params[0] * {input_names[0]} + params[1] * {input_names[1] if len(input_names) > 1 else input_names[0]}
```

Fitness: Your equation will be evaluated by optimizing parameters to minimize MSE on test data.
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial solution with simple linear equation."""
        info = self.dataset_info
        input_names = info['input_cols']

        # Build simple linear combination
        if len(input_names) == 2:
            equation_body = f"    return params[0] * {input_names[0]} + params[1] * {input_names[1]} + params[2]"
            signature = f"{input_names[0]}, {input_names[1]}, params"
        elif len(input_names) == 4:
            terms = [f"params[{i}] * {name}" for i, name in enumerate(input_names)]
            equation_body = f"    return {' + '.join(terms)} + params[{len(input_names)}]"
            signature = ", ".join(input_names) + ", params"
        else:
            terms = [f"params[{i}] * input{i}" for i in range(len(input_names))]
            equation_body = f"    return {' + '.join(terms)} + params[{len(input_names)}]"
            signature = ", ".join([f"input{i}" for i in range(len(input_names))]) + ", params"

        initial_code = f'''import numpy as np

def equation({signature}):
    """Linear baseline model."""
{equation_body}
'''

        # Evaluate the initial solution
        eval_res = self.evaluate_code(initial_code)

        return Solution(
            sol_string=initial_code,
            evaluation_res=eval_res,
            other_info={}
        )
