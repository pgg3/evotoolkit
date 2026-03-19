# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CPU-reviewable tests for the CUDA task shell."""

from evotoolkit.task.cuda_engineering import CudaTask, CudaTaskInfoMaker


def _make_task_info() -> dict:
    return {
        "gpu_type": "RTX 4090",
        "cuda_version": "12.4.1",
        "org_py_code": "def baseline(x):\n    return x\n",
        "func_py_code": "def candidate(x):\n    return x\n",
        "cuda_code": '__global__ void kernel() {}\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}',
        "cuda_info": {
            "runtime": 0.1,
            "prof_string": "baseline profile",
        },
    }


class TestCudaTaskReviewedSurface:
    def test_task_info_maker_supports_fake_mode_without_runtime_dependencies(self):
        task_info = CudaTaskInfoMaker.make_task_info(
            evaluator=None,
            gpu_type="RTX 4090",
            cuda_version="12.4.1",
            org_py_code="def baseline(x):\n    return x\n",
            func_py_code="def candidate(x):\n    return x\n",
            cuda_code="__global__ void kernel() {}",
            fake_mode=True,
        )

        assert task_info["cuda_info"]["runtime"] == 0.1
        assert task_info["cuda_info"]["code"] == "__global__ void kernel() {}"

    def test_task_description_and_initial_solution_use_fake_mode_metadata(self):
        task = CudaTask(_make_task_info(), fake_mode=True)

        description = task.get_base_task_description()
        init_solution = task.make_init_sol_wo_other_info()

        assert "RTX 4090" in description
        assert "CUDA 12.4.1" in description
        assert init_solution.evaluation_res.valid is True
        assert init_solution.evaluation_res.score == -0.1
        assert init_solution.evaluation_res.additional_info["prof_string"] == "baseline profile"

    def test_evaluate_code_returns_fake_result_without_cuda_runtime(self):
        task = CudaTask(_make_task_info(), fake_mode=True)

        result = task.evaluate_code("__global__ void candidate() {}")

        assert result.valid is True
        assert result.score == -0.1
        assert result.additional_info["code"] == "__global__ void candidate() {}"
        assert result.additional_info["compilation_error"] is False
