# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Sandbox execution for CANN operators.

This module provides process isolation for NPU operations to prevent
environment pollution and segmentation faults from exec() calls.

Similar to the CUDA evaluator sandbox pattern.
"""

import multiprocessing as mp
import time
from typing import Any, Dict, Optional, Callable


# ============================================================================
# Worker functions at module level to make them picklable
# ============================================================================

def _verify_correctness_worker(
    python_reference: str,
    context_data: Dict[str, Any],
    device_str: str,
    num_trials: int,
    seed: int,
    project_path: Optional[str],
    return_dict: dict,
    timing_dict: dict,
):
    """Worker for correctness verification in subprocess."""
    try:
        import os
        import torch
        import torch_npu

        # Set up environment for compiled operator (must be done before loading)
        if project_path:
            from pathlib import Path
            opp_path = Path(project_path) / "opp"
            if opp_path.exists():
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(opp_path)
            extension_build = Path(project_path) / "CppExtension" / "build"
            if extension_build.exists():
                lib_path = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = f"{extension_build}:{lib_path}"

        device = torch.device(device_str)

        # Rebuild context from serializable data
        context = {}

        # Execute model_src to get ModelNew
        if "model_src" in context_data:
            exec(context_data["model_src"], context)

        # Execute python_reference to get Model, get_inputs, get_init_inputs
        exec(python_reference, context)

        # Import and run correctness check
        from evotoolkit.task.cann_init.utils.backend.correctness import execute_correctness_check

        passed, error_msg, info = execute_correctness_check(
            context=context,
            device=device,
            synchronize=torch_npu.npu.synchronize,
            num_trials=num_trials,
            seed=seed,
        )

        return_dict["result"] = {
            "pass": passed,
            "error": error_msg if not passed else None,
            **info,
        }
        timing_dict["completed"] = True

    except Exception as e:
        return_dict["result"] = {
            "pass": False,
            "error": f"Sandbox error: {str(e)}",
        }
        timing_dict["completed"] = True


def _measure_performance_worker(
    context_data: Dict[str, Any],
    python_reference: str,
    device_str: str,
    num_warmup: int,
    num_trials: int,
    project_path: Optional[str],
    return_dict: dict,
    timing_dict: dict,
):
    """Worker for performance measurement in subprocess."""
    try:
        import os
        import torch
        import torch_npu

        # Set up environment for compiled operator (must be done before loading)
        if project_path:
            from pathlib import Path
            opp_path = Path(project_path) / "opp"
            if opp_path.exists():
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(opp_path)
            extension_build = Path(project_path) / "CppExtension" / "build"
            if extension_build.exists():
                lib_path = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = f"{extension_build}:{lib_path}"

        device = torch.device(device_str)

        # Rebuild context from serializable data
        context = {}

        # Execute model_src to get ModelNew
        if "model_src" in context_data:
            exec(context_data["model_src"], context)

        # Execute python_reference to get get_inputs, get_init_inputs
        exec(python_reference, context)

        # Import and run performance measurement
        from evotoolkit.task.cann_init.utils.backend.performance import measure_performance

        result = measure_performance(
            context=context,
            device=device,
            synchronize=torch_npu.npu.synchronize,
            event_class=torch_npu.npu.Event,
            num_warmup=num_warmup,
            num_trials=num_trials,
        )

        return_dict["result"] = result
        timing_dict["completed"] = True

    except Exception as e:
        return_dict["result"] = {
            "runtime": None,
            "error": f"Sandbox error: {str(e)}",
        }
        timing_dict["completed"] = True


def _full_evaluate_worker(
    full_code: Dict[str, str],
    op_name: str,
    project_path: str,
    device_str: str,
    python_reference: str,
    num_correctness_trials: int,
    num_perf_trials: int,
    num_warmup: int,
    seed: int,
    skip_correctness: bool,
    skip_performance: bool,
    return_dict: dict,
    timing_dict: dict,
):
    """Worker for full evaluation (compile + correctness + performance) in subprocess."""
    try:
        import torch
        import torch_npu

        device = torch.device(device_str)

        # Import backend functions
        from evotoolkit.task.cann_init.utils.backend import (
            ascend_compile,
            execute_correctness_check,
            measure_performance,
        )

        # Step 1: Compile
        compile_result = ascend_compile(
            full_code=full_code,
            op_name=op_name,
            project_path=project_path,
            device=device_str.replace("npu:", "Ascend910"),  # Convert device string
        )

        if not compile_result["success"]:
            return_dict["result"] = {
                "stage": "compile",
                "success": False,
                "error": compile_result["error"],
            }
            timing_dict["completed"] = True
            return

        context = compile_result["context"]

        # Load python_reference into context
        exec(python_reference, context)

        # Step 2: Verify correctness (unless skipped)
        if not skip_correctness:
            passed, error_msg, info = execute_correctness_check(
                context=context,
                device=device,
                synchronize=torch_npu.npu.synchronize,
                num_trials=num_correctness_trials,
                seed=seed,
            )

            if not passed:
                return_dict["result"] = {
                    "stage": "correctness",
                    "success": False,
                    "error": error_msg,
                    **info,
                }
                timing_dict["completed"] = True
                return

        # Step 3: Measure performance (unless skipped)
        if not skip_performance:
            perf_result = measure_performance(
                context=context,
                device=device,
                synchronize=torch_npu.npu.synchronize,
                event_class=torch_npu.npu.Event,
                num_warmup=num_warmup,
                num_trials=num_perf_trials,
            )

            runtime = perf_result.get("runtime")
            if runtime is None:
                return_dict["result"] = {
                    "stage": "performance",
                    "success": False,
                    "error": perf_result.get("error", "Performance measurement failed"),
                }
                timing_dict["completed"] = True
                return

            return_dict["result"] = {
                "stage": "success",
                "success": True,
                "runtime": runtime,
                "runtime_std": perf_result.get("std"),
            }
        else:
            return_dict["result"] = {
                "stage": "correctness_only" if not skip_correctness else "compile_only",
                "success": True,
            }

        timing_dict["completed"] = True

    except Exception as e:
        import traceback
        return_dict["result"] = {
            "stage": "exception",
            "success": False,
            "error": f"Sandbox error: {str(e)}\n{traceback.format_exc()}",
        }
        timing_dict["completed"] = True


# ============================================================================
# Sandbox Executor Class
# ============================================================================

class CANNSandboxExecutor:
    """
    Sandbox executor for CANN operations using multiprocessing.

    Provides process isolation to prevent:
    - Environment pollution from exec() calls
    - Segmentation faults from NPU memory issues
    - Resource leaks between evaluations

    Usage:
        executor = CANNSandboxExecutor()

        # Full evaluation in sandbox
        result = executor.evaluate_sandbox(
            full_code=full_code,
            op_name="add",
            project_path="/path/to/project",
            device="npu:0",
            python_reference=PYTHON_REF,
        )

        # Just correctness check
        result = executor.verify_correctness_sandbox(
            python_reference=PYTHON_REF,
            context_data={"model_src": model_src},
            device="npu:0",
        )
    """

    def __init__(self, default_timeout: int = 600):
        """
        Initialize sandbox executor.

        Args:
            default_timeout: Default timeout in seconds for operations
        """
        self.default_timeout = default_timeout

    @staticmethod
    def _monitor_process(
        process: mp.Process,
        timing_dict: dict,
        timeout: int,
    ) -> bool:
        """
        Monitor a process with timeout.

        Args:
            process: The subprocess to monitor
            timing_dict: Shared dict for timing info
            timeout: Timeout in seconds

        Returns:
            True if completed normally, False if timeout
        """
        start_time = time.time()

        while process.is_alive():
            if timing_dict.get("completed", False):
                process.join()
                return True

            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                return False

            time.sleep(0.5)

        return True

    @staticmethod
    def _execute_in_sandbox(
        worker_func: Callable,
        worker_args: tuple,
        timeout: int,
        default_error: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a worker function in a sandboxed subprocess.

        Args:
            worker_func: The worker function to execute
            worker_args: Arguments for the worker function
            timeout: Timeout in seconds
            default_error: Default error result if timeout/failure

        Returns:
            Result dict from worker function
        """
        try:
            # Use spawn to ensure clean process
            ctx = mp.get_context("spawn")
            manager = ctx.Manager()
            return_dict = manager.dict()
            timing_dict = manager.dict()

            # Add shared dicts to args
            full_args = worker_args + (return_dict, timing_dict)

            process = ctx.Process(target=worker_func, args=full_args)
            process.start()

            if not CANNSandboxExecutor._monitor_process(process, timing_dict, timeout):
                error_result = default_error.copy()
                error_result["error"] = f"Operation timed out after {timeout}s"
                return error_result

            return dict(return_dict.get("result", default_error))

        except Exception as e:
            error_result = default_error.copy()
            error_result["error"] = f"Sandbox execution error: {str(e)}"
            return error_result

    def verify_correctness_sandbox(
        self,
        python_reference: str,
        context_data: Dict[str, Any],
        device: str = "npu:0",
        num_trials: int = 5,
        seed: int = 1024,
        project_path: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Verify correctness in a sandboxed subprocess.

        Args:
            python_reference: Python reference implementation
            context_data: Dict with model_src and other context data
            device: NPU device string
            num_trials: Number of verification trials
            seed: Random seed
            project_path: Path to compiled operator project (for environment setup)
            timeout: Timeout in seconds (uses default if None)

        Returns:
            {"pass": bool, "error": str or None, ...}
        """
        return self._execute_in_sandbox(
            worker_func=_verify_correctness_worker,
            worker_args=(python_reference, context_data, device, num_trials, seed, project_path),
            timeout=timeout or self.default_timeout,
            default_error={"pass": False, "error": "Unknown error"},
        )

    def measure_performance_sandbox(
        self,
        context_data: Dict[str, Any],
        python_reference: str,
        device: str = "npu:0",
        num_warmup: int = 3,
        num_trials: int = 100,
        project_path: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Measure performance in a sandboxed subprocess.

        Args:
            context_data: Dict with model_src and other context data
            python_reference: Python reference (for get_inputs, etc.)
            device: NPU device string
            num_warmup: Number of warmup iterations
            num_trials: Number of measurement trials
            project_path: Path to compiled operator project (for environment setup)
            timeout: Timeout in seconds (uses default if None)

        Returns:
            {"runtime": float, "std": float, ...}
        """
        return self._execute_in_sandbox(
            worker_func=_measure_performance_worker,
            worker_args=(context_data, python_reference, device, num_warmup, num_trials, project_path),
            timeout=timeout or self.default_timeout,
            default_error={"runtime": None, "error": "Unknown error"},
        )

    def evaluate_sandbox(
        self,
        full_code: Dict[str, str],
        op_name: str,
        project_path: str,
        python_reference: str,
        device: str = "npu:0",
        num_correctness_trials: int = 5,
        num_perf_trials: int = 100,
        num_warmup: int = 3,
        seed: int = 1024,
        skip_correctness: bool = False,
        skip_performance: bool = False,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation (compile + correctness + performance) in sandbox.

        This is the recommended method for full evaluation as it runs
        the entire pipeline in a single subprocess, avoiding any
        environment pollution in the main process.

        Args:
            full_code: Dict with all code components
            op_name: Operator name
            project_path: Path for compilation output
            python_reference: Python reference implementation
            device: NPU device string
            num_correctness_trials: Number of correctness trials
            num_perf_trials: Number of performance trials
            num_warmup: Number of warmup iterations
            seed: Random seed
            skip_correctness: Skip correctness check
            skip_performance: Skip performance measurement
            timeout: Timeout in seconds (uses default if None)

        Returns:
            {
                "stage": "success" | "compile" | "correctness" | "performance" | "exception",
                "success": bool,
                "runtime": float (if success),
                "error": str (if failed),
            }
        """
        return self._execute_in_sandbox(
            worker_func=_full_evaluate_worker,
            worker_args=(
                full_code,
                op_name,
                project_path,
                device,
                python_reference,
                num_correctness_trials,
                num_perf_trials,
                num_warmup,
                seed,
                skip_correctness,
                skip_performance,
            ),
            timeout=timeout or self.default_timeout,
            default_error={
                "stage": "exception",
                "success": False,
                "error": "Unknown error",
            },
        )
