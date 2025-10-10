# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import os
import torch
import tempfile
import torch.utils.cpp_extension as cpp_extension
from torch.profiler import profile, record_function, ProfilerActivity
from .shared_lock import global_file_lock
from .utils import set_seed


def get_py_runtime(py_code: str, timing_dict: dict) -> dict:
    # Phase 1: Pre-lock work (counts toward timeout)
    result_dict = {"runtime": float("inf"), "error_msg": None}
    ns = {}
    exec(py_code, ns)

    # Mark ready to acquire lock
    timing_dict["ready_for_lock"] = True
    # Waiting for lock (does NOT count toward timeout)
    with global_file_lock():
        # Phase 2: Post-lock work (counts toward timeout), Mark lock acquired, resume timeout counting
        timing_dict["lock_acquired"] = True
        set_seed(0)
        init_inputs = ns["get_init_inputs"]()
        init_inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        with torch.no_grad():
            try:
                model_inst = ns["Model"](*init_inputs)
                model_inst = model_inst.cuda()
            except Exception as e:
                result_dict["error_msg"] = f"Failed to create the model: {str(e)}"
                return result_dict
            inputs = ns["get_inputs"]()
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
        # warmup
        for _ in range(3):
            model_inst(*inputs)
            torch.cuda.synchronize()
        run_time_list = []
        for _ in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            model_inst(*inputs)
            end_event.record()

            # Synchronize to ensure the events have completed
            torch.cuda.synchronize()

            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)

            run_time_list.append(elapsed_time_ms)

        cuda_runtime = sum(run_time_list) / 100.0
        result_dict["runtime"] = cuda_runtime
        result_dict["error_msg"] = None
        return result_dict


def get_cuda_runtime(
    func_code: str, cuda_code: str, temp_path: str, temp_str: str, timing_dict: dict
) -> dict:
    # Phase 1: Pre-lock work (counts toward timeout)
    result_dict = {
        "temp_str": temp_str,
        "runtime": float("inf"),
        "error_msg": None,
        "prof_string": None,
    }

    if temp_str is None:
        temp_str = next(tempfile._get_candidate_names())
        result_dict["temp_str"] = temp_str

    use_temp_path = os.path.join(temp_path, temp_str)
    cuda_file_path = os.path.join(use_temp_path, "cuda_code.cu")
    build_path = os.path.join(use_temp_path, "build")
    os.makedirs(build_path, exist_ok=True)
    if not os.path.exists(cuda_file_path):
        with open(cuda_file_path, "w") as f:
            f.write(cuda_code)

    try:
        cuda_fn = cpp_extension.load(
            name=f"op_{temp_str}",
            sources=[cuda_file_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=build_path,
            with_cuda=True,
            verbose=False,
        )
    except Exception as e:
        result_dict["error_msg"] = f"Failed to compile CUDA code: {str(e)}"
        return result_dict

    func_ns = {}
    exec(func_code, func_ns)

    # Mark ready to acquire lock
    timing_dict["ready_for_lock"] = True
    # Waiting for lock (does NOT count toward timeout)
    with global_file_lock():
        # Phase 2: Post-lock work (counts toward timeout), Mark lock acquired, resume timeout counting
        timing_dict["lock_acquired"] = True
        set_seed(0)
        init_inputs = func_ns["get_init_inputs"]()
        init_inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        with torch.no_grad():
            try:
                func_model_inst_copy = func_ns["Model"](*init_inputs)
                func_model_inst_copy = func_model_inst_copy.cuda()
            except Exception as e:
                result_dict["error_msg"] = f"Failed to create the model: {str(e)}"
                return result_dict

            inputs = func_ns["get_inputs"]()
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

        # warmup
        for _ in range(3):
            func_model_inst_copy(*inputs, fn=cuda_fn.forward)
            torch.cuda.synchronize()

        run_time_list = []
        for _ in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            func_model_inst_copy(*inputs, fn=cuda_fn.forward)
            end_event.record()

            # Synchronize to ensure the events have completed
            torch.cuda.synchronize()

            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)

            run_time_list.append(elapsed_time_ms)

        cuda_runtime = sum(run_time_list) / 100.0
        result_dict["runtime"] = cuda_runtime

        # Generate profiling information
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with record_function("model_inference"):
                func_model_inst_copy(*inputs, fn=cuda_fn.forward)

        # Original table format (kept for reference)
        # prof_string = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

        # Convert profiling data to LLM-friendly format with hierarchy
        key_averages = prof.key_averages()
        prof_data = []

        for event in key_averages:
            # Check for different attribute names for CUDA time
            cuda_time = getattr(event, "cuda_time_total", None) or getattr(
                event, "device_time_total", 0
            )
            cpu_time = getattr(event, "cpu_time_total", None) or getattr(
                event, "cpu_time", 0
            )
            self_cuda_time = getattr(event, "self_cuda_time_total", None) or getattr(
                event, "self_device_time_total", cuda_time
            )

            if cuda_time > 0:  # Only include CUDA events
                prof_data.append(
                    {
                        "name": event.key,
                        "cuda_time_us": cuda_time,
                        "cuda_time_ms": cuda_time / 1000.0,
                        "self_cuda_time_us": self_cuda_time,
                        "self_cuda_time_ms": self_cuda_time / 1000.0,
                        "cpu_time_us": cpu_time,
                        "cpu_time_ms": cpu_time / 1000.0,
                        "count": event.count,
                        "input_shapes": str(getattr(event, "input_shapes", []))
                        if hasattr(event, "input_shapes")
                        else None,
                        "cuda_memory_usage": getattr(event, "cuda_memory_usage", None),
                        # Hierarchy information
                        "has_children": hasattr(event, "cpu_children")
                        and len(getattr(event, "cpu_children", [])) > 0,
                        "is_nested": cuda_time
                        != self_cuda_time,  # If total != self, it contains nested operations
                    }
                )

        # Sort by CUDA time and take top 10
        prof_data = sorted(prof_data, key=lambda x: x["cuda_time_us"], reverse=True)[
            :10
        ]

        # Create LLM-friendly description with hierarchy
        prof_string = "CUDA Performance Profile:\n"
        total_cuda_time = sum(item["cuda_time_us"] for item in prof_data)

        for i, item in enumerate(prof_data, 1):
            percentage = (
                (item["cuda_time_us"] / total_cuda_time * 100)
                if total_cuda_time > 0
                else 0
            )

            # Show if operation contains nested children
            if item["is_nested"]:
                self_time = item["self_cuda_time_ms"]
                total_time = item["cuda_time_ms"]
                nested_time = total_time - self_time
                prof_string += f"{i}. {item['name']}: {total_time:.3f}ms total ({percentage:.1f}% of total)\n"
                prof_string += f"   ├─ Self time: {self_time:.3f}ms\n"
                prof_string += f"   └─ Nested operations: {nested_time:.3f}ms\n"
            else:
                prof_string += f"{i}. {item['name']}: {item['cuda_time_ms']:.3f}ms ({percentage:.1f}% of total)\n"

            prof_string += f"   Called {item['count']} times"

            if item["input_shapes"] and item["input_shapes"] != "[]":
                prof_string += f", Input shapes: {item['input_shapes']}"
            prof_string += "\n"

        # Calculate actual non-nested time for more accurate total
        self_times_only = sum(item["self_cuda_time_us"] for item in prof_data)

        prof_string += (
            f"\nTotal CUDA time (including nested): {total_cuda_time / 1000.0:.3f}ms"
        )
        prof_string += (
            f"\nActual execution time (self only): {self_times_only / 1000.0:.3f}ms"
        )
        result_dict["prof_string"] = prof_string

        result_dict["error_msg"] = None
        return result_dict
