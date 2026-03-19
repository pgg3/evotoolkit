# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CPU-reviewable tests for CUDA reference-task interfaces."""

from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.cuda_engineering import (
    CudaTask,
    EoHCudaInterface,
    EvoEngineerFreeCudaInterface,
    EvoEngineerFullCudaInterface,
    EvoEngineerInsightCudaInterface,
    FunSearchCudaInterface,
)


def _make_task() -> CudaTask:
    data = {
        "gpu_type": "RTX 4090",
        "cuda_version": "12.4.1",
        "org_py_code": "def baseline(x):\n    return x\n",
        "func_py_code": "def candidate(x):\n    return x\n",
        "cuda_code": '__global__ void kernel() {}\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}',
        "cuda_info": {
            "runtime": 0.25,
            "prof_string": "baseline profile",
        },
    }
    return CudaTask(data, fake_mode=True)


def _make_solution(score: float = -0.25, name: str = "baseline", thought: str = "shared memory") -> Solution:
    return Solution(
        '__global__ void kernel() {}\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}',
        other_info={"algorithm": "{tiling}", "name": name, "thought": thought},
        evaluation_res=EvaluationResult(
            valid=True,
            score=score,
            additional_info={"prof_string": "profile output"},
        ),
    )


class TestEoHCudaInterface:
    def test_prompt_i1_contains_baseline_code(self):
        iface = EoHCudaInterface(_make_task())

        prompt = iface.get_prompt_i1()

        assert "CUDA kernel code example" in prompt[0]["content"]
        assert "PYBIND11_MODULE" in prompt[0]["content"]

    def test_parse_response_extracts_algorithm_and_code(self):
        iface = EoHCudaInterface(_make_task())

        solution = iface.parse_response("{tile and fuse}\n```cpp\n__global__ void kernel() {}\n```")

        assert solution.other_info["algorithm"] == "{tile and fuse}"
        assert "__global__ void kernel" in solution.sol_string


class TestFunSearchCudaInterface:
    def test_get_prompt_without_solutions_falls_back_to_original_code(self):
        iface = FunSearchCudaInterface(_make_task())

        prompt = iface.get_prompt([])

        assert "original CUDA kernel code" in prompt[0]["content"]
        assert "PYBIND11_MODULE" in prompt[0]["content"]

    def test_parse_response_prefers_code_block(self):
        iface = FunSearchCudaInterface(_make_task())

        solution = iface.parse_response("```cuda\n__global__ void improved() {}\n```")

        assert solution.sol_string == "__global__ void improved() {}"


class TestEvoEngineerFullCudaInterface:
    def test_get_operator_prompt_uses_baseline_when_best_is_missing(self):
        iface = EvoEngineerFullCudaInterface(_make_task())

        prompt = iface.get_operator_prompt("init", [], None, ["Use shared memory"])

        assert "CUDA KERNEL OPTIMIZATION TASK" in prompt[0]["content"]
        assert "Use shared memory" in prompt[0]["content"]
        assert "**Name:** Baseline" in prompt[0]["content"]

    def test_parse_response_uses_fallback_strategies(self):
        iface = EvoEngineerFullCudaInterface(_make_task())

        solution = iface.parse_response("code:\n```cpp\n__global__ void fallback() {}\n```")

        assert solution.other_info["name"] in {"", "extracted"}
        assert "__global__ void fallback" in solution.sol_string

    def test_unknown_operator_raises_value_error(self):
        iface = EvoEngineerFullCudaInterface(_make_task())

        try:
            iface.get_operator_prompt("unknown", [], _make_solution(), [])
        except ValueError as exc:
            assert "Unknown operator" in str(exc)
        else:
            raise AssertionError("Expected ValueError for unknown operator")


class TestEvoEngineerFreeCudaInterface:
    def test_operator_sets_and_prompt_shape(self):
        iface = EvoEngineerFreeCudaInterface(_make_task())

        prompt = iface.get_operator_prompt("init", [], None, [])

        assert [op.name for op in iface.get_init_operators()] == ["init"]
        assert [op.name for op in iface.get_offspring_operators()] == ["init"]
        assert "RESPONSE FORMAT" in prompt[0]["content"]

    def test_parse_response_falls_back_to_raw_content(self):
        iface = EvoEngineerFreeCudaInterface(_make_task())

        solution = iface.parse_response("__global__ void raw_kernel() {}")

        assert solution.other_info["name"] == "raw"
        assert "__global__ void raw_kernel" in solution.sol_string


class TestEvoEngineerInsightCudaInterface:
    def test_insight_interface_exposes_init_only_operator_sets(self):
        iface = EvoEngineerInsightCudaInterface(_make_task())

        assert [op.name for op in iface.get_init_operators()] == ["init"]
        assert [op.name for op in iface.get_offspring_operators()] == ["init"]
