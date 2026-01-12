# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Ascend C Evaluator for operator compilation and evaluation.

This module provides a high-level interface for evaluating Ascend C operators,
using backend utilities adapted from MultiKernelBench.

Design:
- Sandbox-only execution for process isolation
- Supports dynamic project_path per evaluation
- Returns CompileResult for save/load support
- Supports rebuild_context for loading pre-compiled results
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .utils.backend import ascend_compile
from .utils.backend.sandbox import CANNSandboxExecutor
from .data_structures import CompileResult


class AscendCEvaluator:
    """
    Evaluator for Ascend C operators.

    This class provides a simplified interface for:
    1. Compile: Create project, write files, build → CompileResult
    2. Verify: Compare outputs against Python reference (in sandbox)
    3. Measure: Profile operator performance (in sandbox)

    All evaluation (correctness/performance) runs in sandbox subprocess
    to prevent environment pollution and segfaults.

    Usage:
        evaluator = AscendCEvaluator(device="Ascend910B")
        result = evaluator.compile(code, "add", project_path="/path")
        verify_result = evaluator.verify_correctness(python_ref, "add")
        perf_result = evaluator.measure_performance("add", python_ref)
    """

    def __init__(
        self,
        project_path: Optional[str] = None,
        device: str = "Ascend910B",
        num_correctness_trials: int = 5,
        num_perf_trials: int = 100,
        num_warmup: int = 3,
        seed: int = 1024,
        sandbox_timeout: int = 600,
        verbose: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            project_path: Default directory for operator project files (optional)
            device: Target device (e.g., "Ascend910B")
            num_correctness_trials: Number of correctness verification trials
            num_perf_trials: Number of performance measurement trials
            num_warmup: Number of warmup runs before performance measurement
            seed: Random seed for reproducibility
            sandbox_timeout: Timeout in seconds for sandbox operations (default: 600)
            verbose: Whether to print build progress messages
        """
        self.project_path = project_path
        self.device = device
        self.num_correctness_trials = num_correctness_trials
        self.num_perf_trials = num_perf_trials
        self.num_warmup = num_warmup
        self.seed = seed
        self.sandbox_timeout = sandbox_timeout
        self.verbose = verbose

        # Sandbox executor for isolated execution
        self._sandbox = CANNSandboxExecutor(default_timeout=sandbox_timeout)

        # Context from compilation (stores model_src for sandbox)
        self.context = {}

    def compile(
        self,
        full_code: Dict[str, str],
        op_name: str,
        project_path: Optional[str] = None,
        kernel_src: Optional[str] = None,
    ) -> CompileResult:
        """
        Compile and deploy the operator code.

        This combines compile + deploy into a single step, matching
        the MultiKernelBench ascend_compile behavior.

        Args:
            full_code: Dictionary with all code components
            op_name: Operator name (e.g., "add")
            project_path: Override default project_path (for parallel compile)
            kernel_src: Original kernel source (for saving)

        Returns:
            CompileResult with success status and context
        """
        # Use provided path or fall back to instance default
        actual_path = project_path or self.project_path
        if actual_path is None:
            import tempfile
            actual_path = tempfile.mkdtemp(prefix=f"cann_{op_name}_")

        result = ascend_compile(
            full_code=full_code,
            op_name=op_name,
            project_path=actual_path,
            device=self.device,
            verbose=self.verbose,
        )

        # Build CompileResult
        compile_result = CompileResult(
            success=result["success"],
            error=result.get("error"),
            project_path=actual_path,
            op_name=op_name,
            context=result.get("context", {}),
            kernel_src=kernel_src,
            full_code=full_code,
        )

        # Store context for later use (backward compatibility)
        if result["success"]:
            self.context = result["context"]
            # Store model_src for sandbox mode (serializable string)
            if full_code and "model_src" in full_code:
                self.context["__model_src__"] = full_code["model_src"]

        return compile_result

    def rebuild_context(self, compile_result: CompileResult) -> bool:
        """
        Rebuild runtime context from a loaded CompileResult.

        For sandbox execution, we only need to store the model_src string.
        The actual exec will happen in the sandbox subprocess.

        Args:
            compile_result: Loaded CompileResult (with empty context)

        Returns:
            True if context was successfully rebuilt
        """
        if not compile_result.is_loadable():
            return False

        if not compile_result.full_code:
            return False

        model_src = compile_result.full_code.get("model_src")
        if not model_src:
            return False

        try:
            # Set up environment for the compiled operator
            self._setup_opp_environment(compile_result.project_path, compile_result.op_name)

            # Store model_src for sandbox execution
            self.context = {"__model_src__": model_src}
            compile_result.context = self.context
            return True
        except Exception:
            return False

    def _setup_opp_environment(self, project_path: str, op_name: str) -> None:
        """Set up environment variables for compiled operator."""
        import os

        project_dir = Path(project_path)
        op_capital = op_name[0].upper() + op_name[1:] + "Custom"
        opp_path = project_dir / "opp"

        if opp_path.exists():
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(opp_path)

        # Add library path for Python bindings
        extension_build = project_dir / "CppExtension" / "build"
        if extension_build.exists():
            lib_path = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{extension_build}:{lib_path}"

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

        Runs in sandbox subprocess for isolation.

        Args:
            python_reference: Python reference implementation
            op_name: Operator name (unused, for API compatibility)

        Returns:
            {"pass": bool, "error": str or None, ...}
        """
        # Prepare context data for sandbox (serializable parts only)
        context_data = {}
        if self.context.get("__model_src__"):
            context_data["model_src"] = self.context["__model_src__"]
        elif "model_src" in self.context:
            context_data["model_src"] = self.context["model_src"]

        return self._sandbox.verify_correctness_sandbox(
            python_reference=python_reference,
            context_data=context_data,
            device="npu:0",
            num_trials=self.num_correctness_trials,
            seed=self.seed,
            project_path=self.project_path,
            timeout=self.sandbox_timeout,
        )

    def measure_performance(
        self, op_name: str, python_reference: Optional[str] = None  # noqa: ARG002
    ) -> Dict[str, Any]:
        """
        Measure operator performance.

        Runs in sandbox subprocess for isolation.

        Args:
            op_name: Operator name (unused, for API compatibility)
            python_reference: Python reference (needed to get get_inputs)

        Returns:
            {"runtime": float, "std": float, ...}
        """
        # Prepare context data for sandbox (serializable parts only)
        context_data = {}
        if self.context.get("__model_src__"):
            context_data["model_src"] = self.context["__model_src__"]
        elif "model_src" in self.context:
            context_data["model_src"] = self.context["model_src"]

        # python_reference is needed to get get_inputs, get_init_inputs
        if python_reference is None:
            python_reference = self.context.get("__python_reference__", "")

        return self._sandbox.measure_performance_sandbox(
            context_data=context_data,
            python_reference=python_reference,
            device="npu:0",
            num_warmup=self.num_warmup,
            num_trials=self.num_perf_trials,
            project_path=self.project_path,
            timeout=self.sandbox_timeout,
        )

    def cleanup(self):
        """Clean up resources."""
        self.context.clear()
