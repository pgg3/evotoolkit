# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Ascend C Evaluator for operator compilation and evaluation.

This module provides a high-level interface for evaluating Ascend C operators,
using backend utilities adapted from MultiKernelBench.
"""

from typing import Any, Dict

from .backend import ascend_compile, execute_correctness_check, measure_performance


class AscendCEvaluator:
    """
    Evaluator for Ascend C operators.

    This class provides a simplified interface for:
    1. Compile: Create project, write files, build
    2. Deploy: Install operator package, build Python bindings
    3. Verify: Compare outputs against Python reference
    4. Measure: Profile operator performance

    The actual implementation is delegated to backend utilities
    adapted from MultiKernelBench.
    """

    def __init__(
        self,
        project_path: str,
        device: str = "Ascend910B",
        num_correctness_trials: int = 5,
        num_perf_trials: int = 100,
        num_warmup: int = 3,
        seed: int = 1024,
    ):
        """
        Initialize the evaluator.

        Args:
            project_path: Directory for operator project files
            device: Target device (e.g., "Ascend910B")
            num_correctness_trials: Number of correctness verification trials
            num_perf_trials: Number of performance measurement trials
            num_warmup: Number of warmup runs before performance measurement
            seed: Random seed for reproducibility
        """
        self.project_path = project_path
        self.device = device
        self.num_correctness_trials = num_correctness_trials
        self.num_perf_trials = num_perf_trials
        self.num_warmup = num_warmup
        self.seed = seed

        # Lazy import torch_npu
        self._torch = None
        self._torch_npu = None
        self._torch_device = None

        # Context from compilation (stores Model, ModelNew, etc.)
        self.context = {}

    def _init_torch(self):
        """Lazy initialize torch and torch_npu."""
        if self._torch is None:
            import torch
            import torch_npu
            self._torch = torch
            self._torch_npu = torch_npu
            self._torch_device = torch.device("npu:0")

    def compile(self, full_code: Dict[str, str], op_name: str) -> Dict[str, Any]:
        """
        Compile and deploy the operator code.

        This combines compile + deploy into a single step, matching
        the MultiKernelBench ascend_compile behavior.

        Args:
            full_code: Dictionary with all code components
            op_name: Operator name (e.g., "add")

        Returns:
            {"success": bool, "error": str or None}
        """
        result = ascend_compile(
            full_code=full_code,
            op_name=op_name,
            project_path=self.project_path,
            device=self.device,
        )

        # Store context for later use
        if result["success"]:
            self.context = result["context"]

        return {"success": result["success"], "error": result["error"]}

    def deploy(self, op_name: str) -> Dict[str, Any]:  # noqa: ARG002
        """
        Deploy is already done in compile().

        This method exists for API compatibility but does nothing
        since ascend_compile handles both compile and deploy.

        Args:
            op_name: Operator name (unused)

        Returns:
            {"success": True, "error": None}
        """
        # Deploy is already done in compile()
        return {"success": True, "error": None}

    def verify_correctness(
        self, python_reference: str, op_name: str  # noqa: ARG002
    ) -> Dict[str, Any]:
        """
        Verify operator correctness against Python reference.

        Args:
            python_reference: Python reference implementation
            op_name: Operator name (unused, for API compatibility)

        Returns:
            {"pass": bool, "error": str or None, ...}
        """
        self._init_torch()

        # Execute Python reference to get Model class
        try:
            exec(python_reference, self.context)
        except Exception as e:
            return {"pass": False, "error": f"Failed to execute reference: {str(e)}"}

        # Run correctness check
        passed, error_msg, info = execute_correctness_check(
            context=self.context,
            device=self._torch_device,
            synchronize=self._torch_npu.npu.synchronize,
            num_trials=self.num_correctness_trials,
            seed=self.seed,
        )

        result = {"pass": passed, "error": error_msg if not passed else None}
        result.update(info)
        return result

    def measure_performance(self, op_name: str) -> Dict[str, Any]:  # noqa: ARG002
        """
        Measure operator performance.

        Args:
            op_name: Operator name (unused, for API compatibility)

        Returns:
            {"runtime": float, "std": float, ...}
        """
        self._init_torch()

        return measure_performance(
            context=self.context,
            device=self._torch_device,
            synchronize=self._torch_npu.npu.synchronize,
            event_class=self._torch_npu.npu.Event,
            num_warmup=self.num_warmup,
            num_trials=self.num_perf_trials,
        )

    def cleanup(self):
        """Clean up resources."""
        self.context.clear()
        if self._torch_npu is not None:
            self._torch_npu.npu.empty_cache()
            self._torch_npu.npu.synchronize(device=self._torch_device)
