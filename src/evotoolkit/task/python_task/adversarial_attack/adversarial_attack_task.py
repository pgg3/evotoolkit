# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Adversarial Attack Task implementation.

This task evolves the `draw_proposals` function to generate effective
adversarial examples for black-box attacks.
"""

import numpy as np
import types
import sys
import warnings
from typing import Callable, Optional
from evotoolkit.core import Solution, EvaluationResult
from evotoolkit.task.python_task.python_task import PythonTask
from evotoolkit.registry import register_task


@register_task("AdversarialAttack")
class AdversarialAttackTask(PythonTask):
    """
    Adversarial attack task for evolving proposal generation algorithms.

    This task evaluates Python code that defines a `draw_proposals` function
    used in black-box adversarial attacks to generate candidate adversarial examples.

    The evolved function should effectively balance exploration and exploitation
    to find adversarial examples with minimal distortion.
    """

    def __init__(
        self,
        model: Optional[any] = None,
        test_loader: Optional[any] = None,
        attack_steps: int = 1000,
        n_test_samples: int = 10,
        timeout_seconds: float = 300.0,
        use_mock: bool = False
    ):
        """
        Initialize adversarial attack task.

        Args:
            model: Target model to attack (PyTorch model). If None or use_mock=True,
                   uses mock evaluation for testing.
            test_loader: DataLoader with test samples. If None or use_mock=True,
                        uses mock evaluation. Data preprocessing/normalization should
                        be handled externally (e.g., in model wrapper or transform).
            attack_steps: Number of attack iterations per sample
            n_test_samples: Number of test samples to evaluate
            timeout_seconds: Execution timeout
            use_mock: If True, uses mock evaluation (returns random fitness)
        """
        self.model = model
        self.test_loader = test_loader
        self.attack_steps = attack_steps
        self.n_test_samples = n_test_samples
        self.use_mock = use_mock

        # Initialize data
        data = {
            'attack_steps': attack_steps,
            'n_test_samples': n_test_samples,
            'use_mock': use_mock
        }

        super().__init__(data=data, timeout_seconds=timeout_seconds)

    def _process_data(self, data):
        """Process input data and create task_info."""
        self.data = data
        self.task_info = {
            'attack_steps': data['attack_steps'],
            'n_test_samples': data['n_test_samples'],
            'use_mock': data['use_mock']
        }

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """
        Evaluate Python code for adversarial attack.

        The code must define a `draw_proposals` function.
        """
        # If using mock mode, return random fitness
        if self.use_mock:
            return EvaluationResult(
                valid=True,
                score=float(np.random.uniform(1.0, 5.0)),  # Random L2 distance
                additional_info={
                    'avg_distance': float(np.random.uniform(1.0, 5.0)),
                    'mock': True
                }
            )

        # Check dependencies
        try:
            import torch
            import foolbox as fb
        except ImportError as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': f'Missing dependency: {str(e)}. Install with: pip install torch foolbox'}
            )

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
        try:
            exec(candidate_code, namespace)
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': f'Code execution error: {str(e)}'}
            )

        # Check if draw_proposals function exists
        if 'draw_proposals' not in namespace:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Function "draw_proposals" not found in code'}
            )

        draw_proposals_func = namespace['draw_proposals']

        # Evaluate the attack
        try:
            avg_distance = self._evaluate_attack(draw_proposals_func)

            if avg_distance is None or np.isnan(avg_distance) or np.isinf(avg_distance):
                return EvaluationResult(
                    valid=False,
                    score=float('-inf'),
                    additional_info={'error': 'Attack evaluation returned None/NaN/Inf'}
                )

            # Lower distance is better - negate for maximization
            score = -float(avg_distance)

            return EvaluationResult(
                valid=True,
                score=score,
                additional_info={
                    'avg_distance': float(avg_distance),
                    'attack_steps': self.attack_steps
                }
            )

        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': f'Attack evaluation error: {str(e)}'}
            )

    def _evaluate_attack(self, draw_proposals_func: Callable) -> Optional[float]:
        """
        Evaluate attack with evolved draw_proposals function.

        Returns:
            float: Average L2 distance, or None if failed
        """
        import torch
        import foolbox as fb
        import eagerpy as ep

        # Create module with draw_proposals
        heuristic_module = types.ModuleType("heuristic_module")
        heuristic_module.draw_proposals = draw_proposals_func
        sys.modules[heuristic_module.__name__] = heuristic_module

        # Import EvoAttack
        from .evo_attack import EvoAttack

        # Setup model
        if torch.cuda.is_available():
            self.model.cuda()

        # Create foolbox model without preprocessing
        # User should handle normalization externally (in model wrapper or transform)
        fmodel = fb.PyTorchModel(self.model, bounds=(0, 1))

        # Let EvoAttack use its default initial attack (LinearSearchBlendedUniformNoiseAttack)
        # which is a MinimizationAttack. Our improved error handling in evo_attack.py
        # will handle cases where initial attack fails.
        attack = EvoAttack(heuristic_module, init_attack=None, steps=self.attack_steps)

        # Evaluate on test samples
        distances = []
        sample_count = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for x, y in self.test_loader:
                if sample_count >= self.n_test_samples:
                    break

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # Clip to [0, 1] using eagerpy
                x_ep = ep.astensor(x)
                min_, max_ = fmodel.bounds
                x_ep = ep.clip(x_ep, min_, max_)
                x = x_ep.raw

                try:
                    # Run attack with foolbox criterion
                    img_adv = attack.run(fmodel, x, fb.criteria.Misclassification(y))

                    # Calculate L2 distance
                    distance = torch.linalg.norm((x - img_adv).flatten(start_dim=1), axis=1)
                    distance = distance.mean()

                    # Check if distance is valid
                    dist_value = float(distance.cpu().numpy())
                    if np.isnan(dist_value) or np.isinf(dist_value):
                        print(f"Warning: Invalid distance for sample {sample_count}, using penalty value")
                        distances.append(10.0)  # Penalty for invalid result
                    else:
                        distances.append(dist_value)

                except Exception as e:
                    # If attack fails, use a penalty value instead of failing completely
                    print(f"Warning: Attack failed on sample {sample_count}: {str(e)}")
                    distances.append(10.0)  # Penalty distance for failed attack

                sample_count += 1

        if not distances:
            return None

        return float(np.mean(distances))

    def get_base_task_description(self) -> str:
        """Get task description."""
        return """You are an expert in adversarial machine learning and optimization algorithms.

Task: Design an effective proposal generation algorithm for black-box adversarial attacks

Your goal is to evolve a `draw_proposals` function that generates high-quality candidate
adversarial examples to fool a neural network classifier with minimal distortion.

Function Signature:
```python
def draw_proposals(
    org_img: np.ndarray,
    best_adv_img: np.ndarray,
    std_normal_noise: np.ndarray,
    hyperparams: np.ndarray
) -> np.ndarray:
    \"\"\"
    Generate a new candidate adversarial example.

    Args:
        org_img: Original clean image, shape (3, H, W), values in [0, 1]
        best_adv_img: Current best adversarial example, shape (3, H, W), values in [0, 1]
        std_normal_noise: Random normal noise, shape (3, H, W)
        hyperparams: Step size parameter, shape (1,), range [0.5, 1.5]
                    Gets larger when algorithm finds more adversarial examples

    Returns:
        np.ndarray: New candidate adversarial example, shape (3, H, W)
    \"\"\"
```

Requirements:
- All inputs and outputs are numpy arrays
- Output must have same shape as org_img: (3, H, W)
- Output values should stay in [0, 1] (will be clipped automatically)
- Use numpy operations (np.linalg.norm, np.dot, etc.)

Available Operations:
- Arithmetic: +, -, *, /
- Linear algebra: np.dot, np.linalg.norm, np.matmul
- Array operations: .reshape(), .flatten(), etc.

Strategy Guidelines:
1. **Direction**: Move from best_adv_img towards the decision boundary
2. **Step size**: Use hyperparams to control exploration vs exploitation
3. **Noise**: Incorporate std_normal_noise for exploration
4. **Distance**: Consider the vector from org_img to best_adv_img

Key Insights:
- Smaller L2 distance from org_img is better
- The candidate should be adversarial (fool the model)
- Balance between exploitation (refining best_adv_img) and exploration (using noise)
- hyperparams adapts: increases when finding adversarials, decreases otherwise

Example Structure:
```python
import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    # Reshape to vectors
    org = org_img.flatten()
    best = best_adv_img.flatten()
    noise = std_normal_noise.flatten()

    # Compute direction from org to best
    direction = org - best

    # Your algorithm here: combine direction, noise, and hyperparams
    # ...

    # Reshape back to image
    return candidate.reshape(org_img.shape)
```

Fitness: Your function will be evaluated by running attacks and measuring the average L2
distance of adversarial examples from original images. Lower distance = better score.
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial solution with baseline algorithm."""
        initial_code = '''import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """
    Baseline proposal generation using parallel and perpendicular components.

    This is a simple baseline that combines:
    1. Movement along the direction from org to best (parallel)
    2. Random exploration perpendicular to that direction
    """
    # Reshape to flat vectors
    original_shape = org_img.shape
    org = org_img.flatten()
    best = best_adv_img.flatten()
    noise = std_normal_noise.flatten()

    # Compute norms and direction
    noise_norm = np.linalg.norm(noise)
    direction = org - best
    direction_norm = np.linalg.norm(direction)

    # Parallel component (along org->best direction)
    step_size = (noise_norm * hyperparams[0]) ** 2
    d_parallel = step_size * direction

    # Perpendicular component (exploration)
    if direction_norm > 1e-8:
        # Project noise onto direction
        dot_product = np.dot(direction, noise)
        projection = (dot_product / direction_norm) * direction
        # Perpendicular = noise - projection
        d_perpendicular = (projection / direction_norm - direction_norm * noise) * hyperparams[0]
    else:
        d_perpendicular = noise * hyperparams[0]

    # Combine components
    candidate = best + d_parallel + d_perpendicular

    # Reshape back
    return candidate.reshape(original_shape)
'''
        # Evaluate the initial solution
        eval_res = self.evaluate_code(initial_code)

        return Solution(
            sol_string=initial_code,
            evaluation_res=eval_res,
            other_info={}
        )
