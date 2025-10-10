# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
CUDA task class for evolutionary optimization.

This module contains the base class for CUDA-based tasks, unifying
the functionality of CudaEvaluator and CudaTaskConfig.
"""

import tempfile
from typing import Optional, Dict, Any
from evotoolkit.core import BaseTask, Solution, EvaluationResult
from .evaluator import Evaluator


class CudaTaskInfoMaker:
    @classmethod
    def make_task_info(
        cls,
        evaluator: Evaluator,
        gpu_type: str,
        cuda_version: str,
        org_py_code: str,
        func_py_code: str,
        cuda_code: str,
        fake_mode: bool = False,
        **kwargs,
    ) -> dict:
        task_info = {
            "gpu_type": gpu_type,
            "cuda_version": cuda_version,
            "org_py_code": org_py_code,
            "func_py_code": func_py_code,
            "cuda_code": cuda_code,
        }
        # LOCK_FILE = os.path.join(tempfile.gettempdir(), "evotool_cross_process.lock")
        # shutil.rmtree(LOCK_FILE, ignore_errors=True)
        if fake_mode:
            info_dict = {
                "name": "baseline",
                "thought": "baseline",
                "code": cuda_code,
                "temp_str": "xxx",
                "runtime": 0.1,
                "prof_string": "xxx",
                "compilation_error": False,
                "comparison_error": False,
            }
            task_info["cuda_info"] = info_dict
            return task_info
        cuda_info_dict = evaluator.get_cuda_runtime_sandbox(func_py_code, cuda_code)
        info_dict = {
            "name": "baseline",
            "thought": "baseline",
            "code": cuda_code,
            "temp_str": cuda_info_dict["temp_str"],
            "runtime": cuda_info_dict["runtime"],
            "prof_string": cuda_info_dict["prof_string"],
            "compilation_error": False,
            "comparison_error": False,
        }
        task_info["cuda_info"] = info_dict
        return task_info


class CudaTask(BaseTask):
    """
    Base class for CUDA-based evolutionary optimization tasks.

    This class unifies CudaEvaluator and CudaTaskConfig functionality,
    providing a common base for CUDA kernel optimization tasks.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        temp_path: Optional[str] = None,
        fake_mode: bool = False,
    ):
        """
        Initialize the CUDA task with input data.

        Args:
            data: Task-specific input data (task_info dict)
            temp_path: Temporary path for CUDA compilation
            fake_mode: If True, skip actual CUDA evaluation
        """
        self.temp_path = temp_path or tempfile.mkdtemp()
        self.fake_mode = fake_mode
        super().__init__(data)

        self.evaluator = Evaluator(self.temp_path)

    def _process_data(self, data):
        """Process CUDA task data."""
        self.task_info = data
        self.org_py_code = data["org_py_code"]
        self.func_py_code = data["func_py_code"]
        self.cuda_code = data["cuda_code"]

    def get_task_type(self) -> str:
        """Get task type as 'Cuda'."""
        return "Cuda"

    def get_base_task_description(self) -> str:
        """Get the base task description using task info"""
        gpu_type = self.task_info.get("gpu_type", "RTX 4090")
        cuda_version = self.task_info.get("cuda_version", "12.4.1")
        return f"""You are a Machine Learning Engineer trying to reduce the runtime of a CUDA kernel. Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. The kernel will be run on a {gpu_type} GPU with CUDA {cuda_version}.
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial solution from task info."""
        init_sol = Solution(self.task_info["cuda_code"])
        evaluation_res = EvaluationResult(
            valid=True,
            score=-self.task_info["cuda_info"]["runtime"],
            additional_info={
                "code": self.task_info["cuda_code"],
                "temp_str": None,
                "runtime": self.task_info["cuda_info"]["runtime"],
                "prof_string": self.task_info["cuda_info"]["prof_string"],
                "compilation_error": False,
                "comparison_error": False,
                "error_msg": None,
                "exception": None,
            },
        )
        init_sol.evaluation_res = evaluation_res
        return init_sol

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        """Evaluate CUDA kernel code."""

        try:
            if self.fake_mode:
                return EvaluationResult(
                    valid=True,
                    score=-0.1,
                    additional_info={
                        "code": candidate_code,
                        "temp_str": None,
                        "runtime": 0.1,
                        "prof_string": None,
                        "compilation_error": False,
                        "comparison_error": False,
                        "error_msg": None,
                        "exception": None,
                    },
                )

            cuda_comparison_result = self.evaluator.compare_func_cuda_sandbox(
                self.func_py_code, candidate_code
            )

            additional_info = {
                "code": candidate_code,
                "temp_str": cuda_comparison_result.get("temp_str"),
                "runtime": None,
                "prof_string": None,
                "compilation_error": cuda_comparison_result.get(
                    "compilation_error", False
                ),
                "comparison_error": not cuda_comparison_result.get(
                    "correctness", False
                ),
                "error_msg": cuda_comparison_result.get("error_msg", None),
            }

            if cuda_comparison_result.get("correctness", False):
                cuda_runtime_result = self.evaluator.get_cuda_runtime_sandbox(
                    self.func_py_code,
                    candidate_code,
                    cuda_comparison_result.get("temp_str"),
                )
                additional_info["runtime"] = cuda_runtime_result.get("runtime")
                additional_info["prof_string"] = cuda_runtime_result.get("prof_string")

                score = -cuda_runtime_result.get("runtime")
                valid = True
                additional_info["error_msg"] = cuda_runtime_result.get(
                    "error_msg", None
                )
            else:
                score = None
                valid = False

            return EvaluationResult(
                valid=valid, score=score, additional_info=additional_info
            )

        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=None,
                additional_info={
                    "code": candidate_code,
                    "temp_str": None,
                    "runtime": None,
                    "prof_string": None,
                    "compilation_error": True,
                    "comparison_error": True,
                    "error_msg": str(e),
                    "exception": True,
                },
            )
