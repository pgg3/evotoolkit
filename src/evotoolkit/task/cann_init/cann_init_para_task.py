# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Parallel-safe CANNInitTask for concurrent evaluation.

This module provides CANNInitParaTask which extends CANNInitTask with:
- Parallel compilation and correctness verification
- Sequential performance measurement (via class-level lock)

Use this task class when running FunSearch or other methods that
evaluate multiple solutions concurrently.
"""

import threading
from typing import Any, Dict, Optional

from evotoolkit.core import EvaluationResult

from .cann_init_task import CANNInitTask
from .evaluator import AscendCEvaluator
from .data_structures import CANNSolutionConfig


class CANNInitParaTask(CANNInitTask):
    """
    Parallel-safe version of CANNInitTask.

    Inherits all functionality from CANNInitTask but adds a class-level lock
    to serialize performance measurements, preventing NPU resource contention
    when multiple evaluations run concurrently.

    Execution model:
    - Compilation: Parallel (each in isolated temp directory)
    - Correctness verification: Parallel (isolated sandbox processes)
    - Performance measurement: Sequential (protected by _perf_lock)

    Usage:
        task = CANNInitParaTask(
            data={"op_name": "relu", "python_reference": "..."},
            verbose=False  # Hide build info during optimization
        )
        # Use with FunSearch or other concurrent evaluation methods
    """

    # Class-level lock for serializing performance measurements
    # Shared across all instances to prevent concurrent NPU perf tests
    _perf_lock = threading.Lock()

    def _run_verify_and_perf(
        self,
        evaluator: AscendCEvaluator,
        config: CANNSolutionConfig,
        kernel_src: str,
        project_path: str,
        extra_info: Optional[Dict] = None,
    ) -> EvaluationResult:
        """执行正确性验证和性能测量（带并行保护）

        与父类的区别：性能测量阶段使用类级锁串行执行，
        以避免多个NPU性能测试同时运行导致的资源竞争和测量不准确。

        Args:
            evaluator: AscendCEvaluator instance
            config: Solution configuration
            kernel_src: Kernel source code
            project_path: Project directory path
            extra_info: Additional info to include in result

        Returns:
            EvaluationResult with validation status and score
        """
        base_info = {"kernel_src": kernel_src, "project_path": project_path}
        if extra_info:
            base_info.update(extra_info)

        # Correctness check can run in parallel (each in isolated sandbox)
        if not config.skip_correctness:
            verify_result = evaluator.verify_correctness(self.python_reference, self.op_name)
            if not verify_result["pass"]:
                return self._make_result(
                    valid=False,
                    stage="correctness",
                    error=verify_result["error"],
                    python_output=verify_result.get("python_output"),
                    ascend_output=verify_result.get("ascend_output"),
                    max_diff=verify_result.get("max_diff"),
                    **base_info,
                )

        # Performance measurement must be sequential to avoid NPU resource contention
        if not config.skip_performance:
            with CANNInitParaTask._perf_lock:
                perf_result = evaluator.measure_performance(
                    self.op_name, python_reference=self.python_reference
                )
            runtime = perf_result.get("runtime")

            if runtime is None:
                return self._make_result(
                    valid=False,
                    stage="performance",
                    error=perf_result.get("error", "Performance measurement failed"),
                    **base_info,
                )

            return self._make_result(
                valid=True,
                stage="success",
                score=-runtime,
                runtime=runtime,
                runtime_std=perf_result.get("std"),
                **base_info,
            )

        return self._make_result(valid=True, stage="correctness_only", **base_info)

    def get_task_type(self) -> str:
        return "CANNInitPara"
