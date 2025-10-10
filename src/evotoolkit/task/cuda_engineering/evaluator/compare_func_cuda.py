# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import os
import torch
import tempfile
import torch.utils.cpp_extension as cpp_extension
from .shared_lock import global_file_lock

from .utils import set_seed


def compare_func_cuda(
    func_code: str, cuda_code: str, temp_path: str, temp_str: str, timing_dict: dict
) -> dict:
    # Phase 1: Pre-lock work (counts toward timeout)
    result_dict = {
        "temp_str": temp_str,
        "correctness": False,
        "error_msg": None,
        "compilation_error": False,
    }
    if temp_str is None:
        temp_str = next(tempfile._get_candidate_names())
        result_dict["temp_str"] = temp_str

    use_temp_path = os.path.join(temp_path, temp_str)
    cuda_file_path = os.path.join(use_temp_path, "cuda_code.cu")
    build_path = os.path.join(use_temp_path, "build")
    os.makedirs(build_path, exist_ok=True)
    with open(cuda_file_path, "w") as f:
        f.write(cuda_code)

    try:
        # os.environ["TORCH_USE_CUDA_DSA"] = "1"
        cuda_fn = cpp_extension.load(
            name=f"op_{temp_str}",
            sources=[cuda_file_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=build_path,
            with_cuda=True,
            verbose=False,
        )
    except Exception as e:
        result_dict["error_msg"] = str(e)
        result_dict["compilation_error"] = True
        return result_dict
    func_ns = {}
    exec(func_code, func_ns)
    func_copy_ns = {}
    exec(func_code, func_copy_ns)

    # Mark ready to acquire lock
    timing_dict["ready_for_lock"] = True
    # Waiting for lock (does NOT count toward timeout)
    with global_file_lock(timeout=None):
        # Phase 2: Post-lock work (counts toward timeout), Mark lock acquired, resume timeout counting
        timing_dict["lock_acquired"] = True
        init_inputs = func_ns["get_init_inputs"]()
        init_inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        atol = 1e-02
        rtol = 1e-02
        with torch.no_grad():
            set_seed(0)
            func_model_inst = func_ns["Model"](*init_inputs)
            set_seed(0)
            func_model_inst_copy = func_copy_ns["Model"](*init_inputs)

            for i in range(5):
                inputs = func_ns["get_inputs"]()
                inputs = [
                    x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs
                ]

                model = func_model_inst.cuda()
                model_new = func_model_inst_copy.cuda()

                output = model(*inputs)
                torch.cuda.synchronize()
                try:
                    output_new = model_new(*inputs, fn=cuda_fn.forward)
                    torch.cuda.synchronize()
                    if output.shape != output_new.shape:
                        result_dict["error_msg"] = (
                            f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                        return result_dict
                    if not torch.allclose(output, output_new, atol=atol, rtol=rtol):
                        max_diff = torch.max(torch.abs(output - output_new)).item()
                        avg_diff = torch.mean(torch.abs(output - output_new)).item()
                        result_dict["error_msg"] = (
                            f"Output mismatch: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}"
                        )
                        return result_dict
                except Exception as e:
                    result_dict["error_msg"] = f"Error running CUDA code: {e}"
                    return result_dict

        result_dict["correctness"] = True
        return result_dict
