# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import os
import tempfile

import torch.utils.cpp_extension as cpp_extension


def compile_cuda_code(cuda_code: str, temp_path: str) -> dict:
    result_dict = {"temp_str": None,
                   "compile_success": False, "error_msg": None}
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
        cpp_extension.load(
            name=f"op_{temp_str}",
            sources=[cuda_file_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=build_path,
            with_cuda=True,
            verbose=False,
        )
        result_dict["compile_success"] = True
        return result_dict
    except Exception as e:
        result_dict["compile_success"] = False
        result_dict["error_msg"] = str(e)
        return result_dict
