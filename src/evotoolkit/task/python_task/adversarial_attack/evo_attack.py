# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
EvoAttack implementation for adversarial attacks with evolved proposal functions.

Based on L-AutoDA (https://arxiv.org/abs/2401.15335).
"""

from typing import Union, Optional, Any
import numpy as np
import time

try:
    import torch
    import eagerpy as ep
    import foolbox as fb
    from foolbox.devutils import flatten, atleast_kd
    from foolbox.models import Model
    from foolbox.criteria import Criterion
    from foolbox.distances import l2
    from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
    from foolbox.attacks.base import MinimizationAttack
    from foolbox.attacks.base import T
    from foolbox.attacks.base import get_criterion
    from foolbox.attacks.base import get_is_adversarial
    from foolbox.attacks.base import raise_if_kwargs
    from foolbox.attacks.base import verify_input_bounds

    FOOLBOX_AVAILABLE = True
except ImportError:
    FOOLBOX_AVAILABLE = False
    # Create placeholder for type hints
    MinimizationAttack = object
    T = Any


class EvoAttack(MinimizationAttack if FOOLBOX_AVAILABLE else object):
    """
    Evolutionary adversarial attack with custom proposal generation.

    This attack uses an evolved `draw_proposals` function to generate
    candidate adversarial examples.
    """

    if FOOLBOX_AVAILABLE:
        distance = l2

    def __init__(
        self,
        library_module,
        init_attack: Optional[Any] = None,
        steps: int = 1000,
        min_epsilon: float = 0.0,
    ):
        """
        Initialize EvoAttack.

        Args:
            library_module: Module containing draw_proposals function
            init_attack: Initial attack for starting points
            steps: Number of attack iterations
            min_epsilon: Minimum epsilon threshold
        """
        if not FOOLBOX_AVAILABLE:
            raise ImportError(
                "EvoAttack requires foolbox and eagerpy. "
                "Install with: pip install foolbox eagerpy"
            )

        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError("init_attack must be a MinimizationAttack")

        self.library_module = library_module
        self.init_attack = init_attack
        self.steps = steps
        self.min_epsilon = min_epsilon

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        """
        Run the evolutionary attack.

        Args:
            model: Target model
            inputs: Clean input images
            criterion: Attack criterion (target labels)
            early_stop: Early stopping threshold
            starting_points: Initial adversarial examples

        Returns:
            Adversarial examples
        """
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        verify_input_bounds(originals, model)

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        # Initialize starting points
        if starting_points is None:
            init_attack = self.init_attack
            if init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)

            best_advs = init_attack.run(
                model, originals, criterion, early_stop=early_stop
            )
        else:
            best_advs = ep.astensor(starting_points)

        # Setup (moved up to define min_ and max_ before using them)
        N = len(originals)
        ndim = originals.ndim
        min_, max_ = model.bounds
        np_bounds = np.array([min_, max_])

        # Verify starting points are adversarial
        is_adv = is_adversarial(best_advs)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                # If init_attack failed, use noisy version of originals as starting point
                print(f"Warning: init_attack failed for {failed} of {len(is_adv)} inputs. Using noisy originals.")
                noise_scale = 0.1
                noise = ep.normal(originals, shape=originals.shape) * noise_scale
                best_advs = ep.clip(originals + noise, min_, max_)
            else:
                # If user-provided starting points are not adversarial, warn but continue
                print(f"Warning: {failed} of {len(is_adv)} starting_points are not adversarial. Continuing anyway.")

        # Attack parameters
        self.hyperparams = 0.05 * np.ones(originals.shape[0])
        alpha_p = 0.95
        p = np.zeros(originals.shape[0])
        t_start = time.time()

        # Attack loop
        for step in range(1, self.steps + 1):
            originals_raw = originals.raw
            best_advs_raw = best_advs.raw

            orginals_np = originals_raw.cpu().numpy()
            best_advs_np = best_advs_raw.cpu().numpy()

            # Generate candidates using evolved function
            candidates_np = np.zeros(orginals_np.shape)
            standard_noise_np = np.random.normal(size=orginals_np.shape)

            for i in range(orginals_np.shape[0]):
                try:
                    candidate_i = self.library_module.draw_proposals(
                        orginals_np[i],
                        best_advs_np[i],
                        standard_noise_np[i],
                        self.hyperparams[i:i + 1],
                    )
                    candidates_np[i] = candidate_i
                except Exception as e:
                    # If proposal generation fails, keep best_advs
                    candidates_np[i] = best_advs_np[i]

            # Clip to valid range
            candidates_np = np.clip(candidates_np, np_bounds[0], np_bounds[1])

            # Timeout check (2 minutes)
            if time.time() - t_start > 120:
                return restore_type(best_advs)

            # Convert to tensors
            candidates = torch.from_numpy(candidates_np).float().to(originals_raw.device)

            # Check adversarial status
            is_adv = is_adversarial(candidates)

            # Convert to eagerpy
            originals = ep.astensor(originals_raw)
            candidates = ep.astensor(candidates)
            is_adv = ep.astensor(is_adv)
            best_advs = ep.astensor(best_advs_raw)

            # Compute distances
            distances = ep.norms.l2(flatten(originals - candidates), axis=-1)
            source_norms = ep.norms.l2(flatten(originals - best_advs), axis=-1)
            closer = distances < source_norms
            is_best_adv = ep.logical_and(is_adv, closer)
            is_best_adv = atleast_kd(is_best_adv, ndim)

            # Update hyperparameters
            is_best_adv_np = is_best_adv.raw.cpu().numpy()
            is_best_adv_np = is_best_adv_np.astype(float).reshape(-1)
            p = alpha_p * p + (1 - alpha_p) * is_best_adv_np
            self.hyperparams *= np.power(self._f_p(p), 0.1)

            # Update best adversarials
            best_advs = ep.where(is_best_adv, candidates, best_advs)

            # Check convergence
            self.current_epsilons = ep.norms.l2(flatten(best_advs - originals), axis=-1)
            if (self.current_epsilons < self.min_epsilon).all():
                return restore_type(best_advs)

        return restore_type(best_advs)

    def _f_p(self, p: np.ndarray) -> np.ndarray:
        """
        Piecewise linear function for hyperparameter adaptation.

        Maps success rate p to scaling factor:
        - f(0) = 0.5 (reduce step size when failing)
        - f(0.25) = 1.0 (keep step size at 25% success)
        - f(1) = 1.5 (increase step size when succeeding)

        Args:
            p: Success rate in [0, 1]

        Returns:
            Scaling factor for hyperparameters
        """
        l = 0.5
        h = 1.5
        p_threshold = 0.25

        f_p = np.zeros_like(p)
        p_less_idx = p < p_threshold
        p_greater_idx = p >= p_threshold

        f_p[p_less_idx] = l + (1 - l) * p[p_less_idx] / p_threshold
        f_p[p_greater_idx] = 1 + (h - 1) * (p[p_greater_idx] - p_threshold) / (1 - p_threshold)

        return f_p
