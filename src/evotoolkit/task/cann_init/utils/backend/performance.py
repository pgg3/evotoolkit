# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Performance measurement for Ascend C operators.

Adapted from MultiKernelBench/utils/performance.py
"""

import torch
import numpy as np
from typing import Any, Dict, Optional


def _measure_model(
    model,
    inputs,
    device: torch.device,
    synchronize,
    event_class,
    num_warmup: int,
    num_trials: int,
) -> Dict[str, Any]:
    """
    Internal function to measure a single model's performance.

    Args:
        model: Model instance to measure
        inputs: Input tensors
        device: Target device
        synchronize: Synchronization function
        event_class: Event class for timing
        num_warmup: Number of warmup iterations
        num_trials: Number of measurement trials

    Returns:
        Dictionary with runtime statistics
    """
    elapsed_times = []

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            model(*inputs)
            synchronize(device=device)

        # Measure
        for _ in range(num_trials):
            start_event = event_class(enable_timing=True)
            end_event = event_class(enable_timing=True)

            start_event.record()
            model(*inputs)
            end_event.record()

            synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

    return {
        "runtime": float(np.mean(elapsed_times)),
        "std": float(np.std(elapsed_times)),
        "min": float(np.min(elapsed_times)),
        "max": float(np.max(elapsed_times)),
        "num_trials": len(elapsed_times),
    }


def measure_performance(
    context: Dict[str, Any],
    device: torch.device,
    synchronize,
    event_class,
    num_warmup: int = 3,
    num_trials: int = 100,
    eval_target: str = "ModelNew",
    measure_baseline: bool = False,
) -> Dict[str, Any]:
    """
    Measure operator performance using NPU events.

    Adapted from MultiKernelBench time_execution_event_template function.

    Args:
        context: Dictionary containing ModelNew, Model, get_inputs, get_init_inputs
        device: Target device (e.g., torch.device('npu:0'))
        synchronize: Synchronization function (e.g., torch_npu.npu.synchronize)
        event_class: Event class (e.g., torch_npu.npu.Event)
        num_warmup: Number of warmup iterations
        num_trials: Number of measurement trials
        eval_target: Name of model class in context (default: "ModelNew")
        measure_baseline: Whether to also measure baseline (Model) performance

    Returns:
        Dictionary with runtime statistics:
            - runtime: Mean execution time in ms
            - std: Standard deviation
            - min: Minimum time
            - max: Maximum time
            - num_trials: Number of trials
            - baseline_runtime: Baseline (Model) runtime if measure_baseline=True
            - baseline_std: Baseline std if measure_baseline=True
            - speedup: runtime / baseline_runtime if measure_baseline=True
            - error: Error message if failed
    """
    get_inputs = context.get("get_inputs")
    get_init_inputs = context.get("get_init_inputs")
    ModelNew = context.get(eval_target)

    if not all([get_inputs, get_init_inputs, ModelNew]):
        return {"runtime": None, "error": "Missing required functions"}

    try:
        # Prepare inputs
        inputs = get_inputs()
        inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        init_inputs = get_init_inputs()
        init_inputs = [
            x.to(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]

        # Measure ModelNew (our generated kernel)
        custom_model = ModelNew(*init_inputs).to(device)
        result = _measure_model(
            model=custom_model,
            inputs=inputs,
            device=device,
            synchronize=synchronize,
            event_class=event_class,
            num_warmup=num_warmup,
            num_trials=num_trials,
        )
        result["error"] = None

        # Optionally measure baseline (Model from python_reference)
        if measure_baseline:
            Model = context.get("Model")
            if Model is not None:
                try:
                    baseline_model = Model(*init_inputs).to(device)
                    baseline_result = _measure_model(
                        model=baseline_model,
                        inputs=inputs,
                        device=device,
                        synchronize=synchronize,
                        event_class=event_class,
                        num_warmup=num_warmup,
                        num_trials=num_trials,
                    )
                    result["baseline_runtime"] = baseline_result["runtime"]
                    result["baseline_std"] = baseline_result["std"]

                    # Calculate speedup (baseline / ours, >1 means we're faster)
                    if baseline_result["runtime"] > 0:
                        result["speedup"] = baseline_result["runtime"] / result["runtime"]
                    else:
                        result["speedup"] = None
                except Exception as e:
                    result["baseline_error"] = f"Baseline measurement failed: {str(e)}"
            else:
                result["baseline_error"] = "Model class not found in context"

        return result

    except Exception as e:
        return {"runtime": None, "error": f"Performance measurement failed: {str(e)}"}
