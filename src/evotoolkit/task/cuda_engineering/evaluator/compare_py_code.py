# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import torch

from .shared_lock import global_file_lock
from .utils import set_seed


def compare_py_code(org_code: str, func_code: str, timing_dict: dict) -> dict:
    # Phase 1: Pre-lock work (counts toward timeout)
    result_dict = {"correctness": False, "error_msg": None}
    org_ns = {}
    exec(org_code, org_ns)
    func_ns = {}
    exec(func_code, func_ns)

    # Mark ready to acquire lock
    timing_dict["ready_for_lock"] = True
    # Waiting for lock (does NOT count toward timeout)
    with global_file_lock():
        # Phase 2: Post-lock work (counts toward timeout), Mark lock acquired, resume timeout counting
        timing_dict["lock_acquired"] = True
        set_seed(0)
        init_inputs = org_ns["get_init_inputs"]()
        init_inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        with torch.no_grad():
            set_seed(0)
            org_model_inst = org_ns["Model"](*init_inputs)
            try:
                set_seed(0)
                func_model_inst = func_ns["Model"](*init_inputs)
            except Exception as e:
                result_dict["correctness"] = False
                result_dict["error_msg"] = (
                    f"Failed to create the model from the functional code: {str(e)}"
                )
                return result_dict

        atol = 1e-02
        rtol = 1e-02
        for i in range(5):
            with torch.no_grad():
                inputs = org_ns["get_inputs"]()
                inputs = [
                    x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs
                ]

                model = org_model_inst.cuda()
                func_model = func_model_inst.cuda()

                output = model(*inputs)
                torch.cuda.synchronize()

                try:
                    output_new = func_model(*inputs)
                    torch.cuda.synchronize()
                    if output.shape != output_new.shape:
                        result_dict["correctness"] = False
                        result_dict["error_msg"] = (
                            f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                        return result_dict
                    if not torch.allclose(
                        output, output_new, atol=atol, rtol=rtol
                    ):  # fail
                        max_diff = torch.max(torch.abs(output - output_new)).item()
                        avg_diff = torch.mean(torch.abs(output - output_new)).item()
                        result_dict["correctness"] = False
                        result_dict["error_msg"] = (
                            f"Output mismatch: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}"
                        )
                        return result_dict
                except Exception as e:
                    result_dict["correctness"] = False
                    result_dict["error_msg"] = (
                        f"Error running the functional model: {str(e)}"
                    )
                    return result_dict
        result_dict["correctness"] = True
        timing_dict["completed"] = True
        return result_dict
