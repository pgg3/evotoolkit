# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Ascend C Evaluator for operator code compilation and evaluation.

This module implements the evaluator that compiles, deploys, and verifies
Ascend C operators. It adapts the evaluation logic from MultiKernelBench
to work with the evotoolkit framework.
"""

import os
import subprocess
import shutil
from typing import Dict, Any, List
import torch
import torch_npu


class AscendCEvaluator:
    """
    Evaluator for Ascend C operators.

    This class handles:
    1. Compilation: msopgen project creation + build.sh execution
    2. Deployment: Installing the operator package
    3. Correctness verification: Comparing outputs with Python reference
    4. Performance measurement: Timing the operator execution
    """

    def __init__(
        self,
        project_path: str,
        device: str = "Ascend910B",
        num_correctness_trials: int = 3,
        num_perf_trials: int = 10,
        num_warmup: int = 5,
        seed: int = 42,
    ):
        """
        Initialize the evaluator.

        Args:
            project_path: Path to store operator projects
            device: Target device (e.g., "Ascend910B")
            num_correctness_trials: Number of correctness verification trials
            num_perf_trials: Number of performance measurement trials
            num_warmup: Number of warmup runs before performance measurement
            seed: Random seed for reproducibility
        """
        self.project_path = project_path
        self.device = device
        self.num_correctness_trials = num_correctness_trials
        self.num_perf_trials = num_perf_trials
        self.num_warmup = num_warmup
        self.seed = seed

        # Initialize NPU device
        self.torch_device = torch.device('npu:0')

        # Context for storing compiled code
        self.context = {}

        # Track current operator
        self.current_op = None
        self.current_op_capital = None

    def _underscore_to_pascalcase(self, name: str) -> str:
        """Convert underscore_case to PascalCase."""
        return ''.join(word.capitalize() for word in name.split('_'))

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def compile(self, full_code: Dict[str, str], op_name: str) -> Dict[str, Any]:
        """
        Compile the operator code.

        This method:
        1. Executes the generated code to populate context
        2. Creates operator project using msopgen
        3. Writes source files to project directories
        4. Executes build.sh to compile the operator

        Args:
            full_code: Dictionary containing all code components:
                - project_json_src: Operator project JSON
                - host_tiling_src: Host tiling header code
                - host_operator_src: Host operator implementation
                - kernel_src: Kernel implementation
                - python_bind_src: Python binding code
                - model_src: Model code for testing
            op_name: Name of the operator (e.g., "add", "layer_norm")

        Returns:
            Dictionary with:
                - success (bool): Whether compilation succeeded
                - error (str): Error message if failed
        """
        try:
            # Add _custom suffix to operator name
            self.current_op = op_name + '_custom'
            self.current_op_capital = self._underscore_to_pascalcase(self.current_op)

            target_directory = os.path.join(self.project_path, self.current_op_capital)

            # Step 1: Store full_code in context
            self.context = full_code.copy()

            print(f"[INFO] Begin compile for operator: {self.current_op}")

            # Step 2: Create operator project using msopgen
            # Remove existing project if it exists
            if os.path.exists(target_directory):
                print("[INFO] Operator project already exists, deleting...")
                shutil.rmtree(target_directory)

            # Write project JSON
            json_path = os.path.join(self.project_path, f'{self.current_op}.json')
            with open(json_path, 'w') as f:
                f.write(self.context.get('project_json_src', ''))

            # Run msopgen to create project
            print("[INFO] Creating operator project with msopgen...")
            original_cwd = os.getcwd()
            os.chdir(self.project_path)

            try:
                result = subprocess.run(
                    ["msopgen", 'gen', '-i', f'{self.current_op}.json',
                     '-c', self.device, '-lan', 'cpp', '-out', self.current_op_capital],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                print("[INFO] Operator project created successfully")
            except subprocess.CalledProcessError as e:
                error_msg = f"msopgen failed:\nExit Code: {e.returncode}\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}"
                return {"success": False, "error": error_msg}
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "msopgen timed out"}
            finally:
                os.chdir(original_cwd)

            # Step 3: Write source files to project
            print("[INFO] Writing source files...")

            # Write tiling header
            tiling_path = os.path.join(target_directory, 'op_host', f'{self.current_op}_tiling.h')
            with open(tiling_path, 'w') as f:
                f.write(self.context.get('host_tiling_src', ''))

            # Write host operator
            host_op_path = os.path.join(target_directory, 'op_host', f'{self.current_op}.cpp')
            with open(host_op_path, 'w') as f:
                f.write(self.context.get('host_operator_src', ''))

            # Write kernel
            kernel_path = os.path.join(target_directory, 'op_kernel', f'{self.current_op}.cpp')
            with open(kernel_path, 'w') as f:
                f.write(self.context.get('kernel_src', ''))

            # Write Python binding
            cpp_ext_path = os.path.join(self.project_path, 'CppExtension', 'csrc', 'op.cpp')
            os.makedirs(os.path.dirname(cpp_ext_path), exist_ok=True)
            with open(cpp_ext_path, 'w') as f:
                f.write(self.context.get('python_bind_src', ''))

            # Step 4: Build the operator
            print("[INFO] Building operator...")

            # Remove ASCEND_CUSTOM_OPP_PATH to avoid conflicts during build
            os.environ.pop('ASCEND_CUSTOM_OPP_PATH', None)

            os.chdir(target_directory)
            try:
                result = subprocess.run(
                    ["./build.sh"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                print("[INFO] Build succeeded")
            except subprocess.CalledProcessError as e:
                # Extract error lines
                error_lines = []
                for line in e.stdout.split('\n') + e.stderr.split('\n'):
                    if '[ERROR]' in line or 'error:' in line:
                        error_lines.append(line)

                error_msg = f"Build failed:\nExit Code: {e.returncode}\nErrors:\n" + '\n'.join(error_lines)
                return {"success": False, "error": error_msg}
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Build timed out"}
            finally:
                os.chdir(original_cwd)

            return {"success": True, "error": None}

        except Exception as e:
            return {"success": False, "error": f"Unexpected error during compilation: {str(e)}"}

    def deploy(self, op_name: str) -> Dict[str, Any]:
        """
        Deploy the compiled operator.

        This method:
        1. Installs the operator package
        2. Builds Python bindings
        3. Sets up environment variables

        Args:
            op_name: Name of the operator

        Returns:
            Dictionary with:
                - success (bool): Whether deployment succeeded
                - error (str): Error message if failed
        """
        try:
            op_with_suffix = op_name + '_custom'
            op_capital = self._underscore_to_pascalcase(op_with_suffix)
            target_directory = os.path.join(self.project_path, op_capital)

            original_cwd = os.getcwd()

            # Step 1: Deploy operator package
            print("[INFO] Deploying operator package...")
            build_out_dir = os.path.join(target_directory, 'build_out')
            os.chdir(build_out_dir)

            try:
                result = subprocess.run(
                    ["./custom_opp_ubuntu_aarch64.run"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                print("[INFO] Deployment succeeded")
            except subprocess.CalledProcessError as e:
                error_msg = f"Deployment failed:\nExit Code: {e.returncode}\nOutput:\n{e.stdout}"
                return {"success": False, "error": error_msg}
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Deployment timed out"}
            finally:
                os.chdir(original_cwd)

            # Step 2: Build Python bindings
            print("[INFO] Building Python bindings...")
            cpp_ext_dir = os.path.join(self.project_path, 'CppExtension')
            os.chdir(cpp_ext_dir)

            try:
                result = subprocess.run(
                    ['bash', "build_and_run.sh"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                print("[INFO] Python binding succeeded")
            except subprocess.CalledProcessError as e:
                error_msg = f"Python binding failed:\nExit Code: {e.returncode}\nOutput:\n{e.stdout}"
                return {"success": False, "error": error_msg}
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Python binding timed out"}
            finally:
                os.chdir(original_cwd)

            # Step 3: Set up environment variables
            custom_opp_path = f"{self.project_path}/opp/vendors/customize"
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path

            # Update LD_LIBRARY_PATH
            if 'opp/vendors/customize' not in os.environ.get("LD_LIBRARY_PATH", ""):
                custom_lib_path = f"{custom_opp_path}/op_api/lib/"
                existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = f"{custom_lib_path}:{existing_ld_path}"

            # Step 4: Load model code
            try:
                model_src = self.context.get('model_src', '')
                compile(model_src, "<string>", "exec")
                exec(model_src, self.context)
            except Exception as e:
                return {"success": False, "error": f"Failed to load model code: {str(e)}"}

            return {"success": True, "error": None}

        except Exception as e:
            return {"success": False, "error": f"Unexpected error during deployment: {str(e)}"}

    def verify_correctness(
        self,
        python_reference: str,
        op_name: str
    ) -> Dict[str, Any]:
        """
        Verify operator correctness against Python reference.

        This method:
        1. Executes Python reference implementation
        2. Runs both reference and custom models
        3. Compares outputs for shape and value match

        Args:
            python_reference: Python reference implementation code
            op_name: Name of the operator

        Returns:
            Dictionary with:
                - pass (bool): Whether correctness check passed
                - error (str): Error message if failed
                - python_output (optional): Reference output
                - ascend_output (optional): Custom operator output
                - max_diff (optional): Maximum difference between outputs
        """
        try:
            # Execute Python reference
            print("[INFO] Executing Python reference implementation...")
            try:
                exec(python_reference, self.context)
            except Exception as e:
                return {
                    "pass": False,
                    "error": f"Failed to compile reference model: {str(e)}"
                }

            # Get required functions/classes from context
            get_inputs = self.context.get('get_inputs')
            get_init_inputs = self.context.get('get_init_inputs')
            Model = self.context.get('Model')
            ModelNew = self.context.get('ModelNew')

            if not all([get_inputs, get_init_inputs, Model, ModelNew]):
                return {
                    "pass": False,
                    "error": "Missing required functions in context (get_inputs, get_init_inputs, Model, ModelNew)"
                }

            print("[INFO] Verifying correctness...")

            # Initialize models
            init_inputs = get_init_inputs()
            init_inputs = [
                x.to(device=self.torch_device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            with torch.no_grad():
                self._set_seed(self.seed)
                original_model = Model(*init_inputs).to(self.torch_device)
                torch_npu.npu.synchronize(device=self.torch_device)

                self._set_seed(self.seed)
                custom_model = ModelNew(*init_inputs).to(self.torch_device)
                torch_npu.npu.synchronize(device=self.torch_device)

            # Run correctness trials
            with torch.no_grad():
                for trial in range(self.num_correctness_trials):
                    inputs = get_inputs()
                    inputs = [
                        x.to(self.torch_device) if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    ]

                    torch_npu.npu.synchronize(device=self.torch_device)
                    ref_output = original_model(*inputs)
                    torch_npu.npu.synchronize(device=self.torch_device)

                    new_output = custom_model(*inputs)
                    torch_npu.npu.synchronize(device=self.torch_device)

                    # Check shape
                    if ref_output.shape != new_output.shape:
                        return {
                            "pass": False,
                            "error": f"Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}",
                            "python_output": ref_output.shape,
                            "ascend_output": new_output.shape
                        }

                    # Check values
                    if not torch.allclose(ref_output, new_output, atol=1e-02, rtol=1e-02):
                        max_diff = torch.max(torch.abs(ref_output - new_output)).item()
                        return {
                            "pass": False,
                            "error": f"Output value mismatch (max diff: {max_diff:.6f})",
                            "python_output": ref_output.cpu().numpy().tolist()[:10],  # First 10 elements
                            "ascend_output": new_output.cpu().numpy().tolist()[:10],
                            "max_diff": max_diff
                        }

            print("[INFO] Correctness verification passed")
            return {"pass": True, "error": None}

        except Exception as e:
            return {
                "pass": False,
                "error": f"Runtime error during correctness verification: {str(e)}"
            }

    def measure_performance(self, op_name: str) -> Dict[str, Any]:
        """
        Measure operator performance.

        This method:
        1. Warms up the operator
        2. Measures execution time over multiple trials
        3. Returns timing statistics

        Args:
            op_name: Name of the operator

        Returns:
            Dictionary with:
                - runtime (float): Mean execution time in milliseconds
                - std (float): Standard deviation
                - min (float): Minimum execution time
                - max (float): Maximum execution time
                - num_trials (int): Number of measurement trials
        """
        try:
            print("[INFO] Measuring performance...")

            get_inputs = self.context.get('get_inputs')
            get_init_inputs = self.context.get('get_init_inputs')
            ModelNew = self.context.get('ModelNew')

            if not all([get_inputs, get_init_inputs, ModelNew]):
                return {"runtime": None, "error": "Missing required functions in context"}

            # Prepare inputs
            inputs = get_inputs()
            inputs = [
                x.to(self.torch_device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            init_inputs = get_init_inputs()
            init_inputs = [
                x.to(device=self.torch_device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Initialize model
            with torch.no_grad():
                custom_model = ModelNew(*init_inputs).to(self.torch_device)

            elapsed_times = []

            with torch.no_grad():
                # Warmup
                for _ in range(self.num_warmup):
                    custom_model(*inputs)
                    torch_npu.npu.synchronize(device=self.torch_device)

                # Measure
                for _ in range(self.num_perf_trials):
                    start_event = torch_npu.npu.Event(enable_timing=True)
                    end_event = torch_npu.npu.Event(enable_timing=True)

                    start_event.record()
                    custom_model(*inputs)
                    end_event.record()

                    torch_npu.npu.synchronize(device=self.torch_device)

                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    elapsed_times.append(elapsed_time_ms)

            # Calculate statistics
            import numpy as np
            mean_time = float(np.mean(elapsed_times))
            std_time = float(np.std(elapsed_times))
            min_time = float(np.min(elapsed_times))
            max_time = float(np.max(elapsed_times))

            print(f"[INFO] Performance measurement completed: {mean_time:.3f}ms (±{std_time:.3f}ms)")

            return {
                "runtime": mean_time,
                "std": std_time,
                "min": min_time,
                "max": max_time,
                "num_trials": len(elapsed_times)
            }

        except Exception as e:
            return {"runtime": None, "error": f"Performance measurement failed: {str(e)}"}

    def cleanup(self):
        """Clean up resources and cache."""
        self.context.clear()
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize(device=self.torch_device)
