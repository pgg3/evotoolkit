# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from .compare_py_code import compare_py_code
from .compile_cuda_code import compile_cuda_code
from .compare_func_cuda import compare_func_cuda
from .get_runtime import get_py_runtime, get_cuda_runtime

import multiprocessing as mp
from .shared_lock import global_file_lock
import time


# Worker functions at module level to make them picklable
def _compare_py_code_worker(org_code, func_code, return_dict, timing_dict):
    try:
        result_dict = compare_py_code(org_code, func_code, timing_dict)
        return_dict['result'] = result_dict
    except Exception as e:
        return_dict['result'] = {
            "correctness": False,
            "error_msg": f"Error in sandboxed comparison: {str(e)}"
        }


def _compare_func_cuda_worker(func_code, cuda_code, temp_path, temp_str, return_dict, timing_dict):
    try:
        result_dict = compare_func_cuda(func_code, cuda_code, temp_path, temp_str, timing_dict)
        return_dict['result'] = result_dict
    except Exception as e:
        return_dict['result'] = {
            "temp_str": temp_str,
            "correctness": False,
            "error_msg": f"Error in sandboxed CUDA comparison: {str(e)}",
            "compilation_error": True
        }


def _compile_cuda_code_worker(cuda_code, temp_path, return_dict, timing_dict):
    try:
        # Skip lock phases, go directly to execution phase
        timing_dict['ready_for_lock'] = True
        timing_dict['lock_acquired'] = True
        
        result_dict = compile_cuda_code(cuda_code, temp_path)
        return_dict['result'] = result_dict
        timing_dict['completed'] = True
    except Exception as e:
        return_dict['result'] = {
            "temp_str": None,
            "compile_success": False,
            "error_msg": f"Error in sandboxed compilation: {str(e)}"
        }

def _get_py_runtime_worker(py_code, return_dict, timing_dict):
    try:
        result_dict = get_py_runtime(py_code, timing_dict)
        return_dict['result'] = result_dict
        timing_dict['completed'] = True
    except Exception as e:
        return_dict['result'] = {
            "runtime": float("inf"),
            "error_msg": f"Error in sandboxed runtime measurement: {str(e)}"
        }
        timing_dict['completed'] = True


def _get_cuda_runtime_worker(func_code, cuda_code, temp_path, temp_str, return_dict, timing_dict):
    try:
        result_dict = get_cuda_runtime(func_code, cuda_code, temp_path, temp_str, timing_dict)
        return_dict['result'] = result_dict
        timing_dict['completed'] = True
    except Exception as e:
        return_dict['result'] = {
            "temp_str": temp_str,
            "runtime": float("inf"),
            "error_msg": f"Error in sandboxed CUDA runtime measurement: {str(e)}",
            "prof_string": None
        }
        timing_dict['completed'] = True



class Evaluator:
    def __init__(self, temp_path):
        self.temp_path = temp_path

    @staticmethod
    def compare_py_code_sandbox(org_code: str, func_code: str, execution_timeout: int = 300) -> dict:
        return Evaluator.execute_with_phase_timeout(
            _compare_py_code_worker, 
            (org_code, func_code),
            execution_timeout,
            "Comparison timed out after {timeout}s execution time.",
            default_error_result={"correctness": False, "error_msg": "Unknown error"}
        )
    @staticmethod
    def get_py_runtime_sandbox(py_code: str, execution_timeout: int = 300) -> dict:
        return Evaluator.execute_with_phase_timeout(
            _get_py_runtime_worker,
            (py_code,),
            execution_timeout,
            "Python runtime measurement timed out after {timeout}s execution time.",
            default_error_result={"runtime": float("inf"), "error_msg": "Unknown error"}
        )

    def get_cuda_runtime_sandbox(self, func_code: str, cuda_code: str, temp_str: str = None, execution_timeout: int = 300) -> dict:
        return Evaluator.execute_with_phase_timeout(
            _get_cuda_runtime_worker,
            (func_code, cuda_code, self.temp_path, temp_str),
            execution_timeout,
            "CUDA runtime measurement timed out after {timeout}s execution time.",
            default_error_result={"temp_str": temp_str, "runtime": float("inf"), "error_msg": "Unknown error", "prof_string": None}
        )

    def compare_func_cuda_sandbox(self, func_code: str, cuda_code: str, temp_str: str=None, execution_timeout: int = 300) -> dict:
        return Evaluator.execute_with_phase_timeout(
            _compare_func_cuda_worker,
            (func_code, cuda_code, self.temp_path, temp_str),
            execution_timeout,
            "CUDA comparison timed out after {timeout}s execution time.",
            default_error_result={"temp_str": temp_str, "correctness": False, "error_msg": "Unknown error", "compilation_error": True}
        )
    def compile_cuda_code_sandbox(self, cuda_code: str, execution_timeout: int = 300) -> dict:
        return Evaluator.execute_with_phase_timeout(
            _compile_cuda_code_worker,
            (cuda_code, self.temp_path),
            execution_timeout,
            "CUDA compilation timed out after {timeout}s execution time.",
            default_error_result={"temp_str": None, "compile_success": False, "error_msg": "Unknown error"}
        )
    @staticmethod
    def monitor_process_with_phase_timeout(process, timing_dict, execution_timeout, timeout_error_result):
        """
        Monitor a process with timeout only counting actual execution time (excluding lock wait time)
        
        Args:
            process: The multiprocessing.Process to monitor
            timing_dict: Manager dict containing phase flags
            execution_timeout: Timeout in seconds for execution time (excluding lock wait)
            timeout_error_result: Result to return if timeout occurs
            
        Returns:
            True if process completed normally, False if timeout occurred
        """
        execution_time = 0.0
        last_check = time.time()
        waiting_for_lock = False

        while process.is_alive():
            current_time = time.time()

            # Check current phase
            if timing_dict.get('completed', False):
                process.join()
                return True
            elif timing_dict.get('lock_acquired', False) and waiting_for_lock:
                # Just acquired lock, resume counting
                waiting_for_lock = False
                last_check = current_time
            elif timing_dict.get('ready_for_lock', False) and not waiting_for_lock:
                # About to wait for lock, pause counting
                execution_time += current_time - last_check
                waiting_for_lock = True
            elif not waiting_for_lock:
                # Normal execution, count time
                execution_time += current_time - last_check
                last_check = current_time

            # Check timeout only for execution time (not lock wait time)
            if execution_time > execution_timeout:
                # Give process a chance to cleanup locks gracefully
                process.terminate()
                # Wait up to 5 seconds for graceful shutdown
                process.join(timeout=5)
                if process.is_alive():
                    # Force kill if still alive
                    process.kill()
                    process.join()
                timeout_error_result['execution_time'] = execution_time
                return False

            time.sleep(5)

        # Process ended naturally
        if process.is_alive():
            # Give process a chance to cleanup locks gracefully
            process.terminate()
            # Wait up to 5 seconds for graceful shutdown
            process.join(timeout=5)
            if process.is_alive():
                # Force kill if still alive
                process.kill()
                process.join()
        return True

    @staticmethod
    def execute_with_phase_timeout(worker_func, worker_args, execution_timeout, timeout_msg_template, default_error_result=None):
        """
        Execute a worker function with phase-based timeout monitoring
        
        Args:
            worker_func: The worker function to execute
            worker_args: Arguments for the worker function (should include timing_dict)
            execution_timeout: Timeout in seconds for execution time (excluding lock wait)
            timeout_msg_template: Template for timeout error message (should have {timeout} placeholder)
            default_error_result: Default result dict when no result is returned (optional)
            
        Returns:
            Result dict from worker function, or timeout error dict
        """
        # Set spawn method for CUDA compatibility
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        return_dict = manager.dict()
        timing_dict = manager.dict()

        # Add return_dict and timing_dict to worker args
        full_args = worker_args + (return_dict, timing_dict)
        p = mp.Process(target=worker_func, args=full_args)
        p.start()
            
        timeout_error_result = default_error_result.copy()
        timeout_error_result["error_msg"] = timeout_msg_template.format(timeout=execution_timeout)

        if not Evaluator.monitor_process_with_phase_timeout(p, timing_dict, execution_timeout, timeout_error_result):
            return timeout_error_result

        return return_dict.get('result', default_error_result)

