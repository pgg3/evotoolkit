#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Unit tests for AscendCEvaluator.

This test file verifies the basic functionality of each method in the evaluator
without requiring actual NPU hardware or complete operator compilation.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evotoolkit.task.cann_init.evaluator import AscendCEvaluator


class TestAscendCEvaluatorUnit(unittest.TestCase):
    """Unit tests for AscendCEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.evaluator = AscendCEvaluator(
            project_path=self.test_dir,
            device="Ascend910B",
            num_correctness_trials=2,
            num_perf_trials=5,
            num_warmup=2,
            seed=42
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.project_path, self.test_dir)
        self.assertEqual(self.evaluator.device, "Ascend910B")
        self.assertEqual(self.evaluator.num_correctness_trials, 2)
        self.assertEqual(self.evaluator.num_perf_trials, 5)
        self.assertEqual(self.evaluator.num_warmup, 2)
        self.assertEqual(self.evaluator.seed, 42)
        self.assertIsInstance(self.evaluator.context, dict)

    def test_underscore_to_pascalcase(self):
        """Test underscore to PascalCase conversion."""
        self.assertEqual(
            self.evaluator._underscore_to_pascalcase("add_custom"),
            "AddCustom"
        )
        self.assertEqual(
            self.evaluator._underscore_to_pascalcase("layer_norm_custom"),
            "LayerNormCustom"
        )
        self.assertEqual(
            self.evaluator._underscore_to_pascalcase("relu"),
            "Relu"
        )

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    @patch('os.chdir')
    def test_compile_success(self, mock_chdir, mock_rmtree, mock_exists,
                            mock_file, mock_subprocess):
        """Test successful compilation."""
        # Mock file system
        mock_exists.return_value = False

        # Mock subprocess calls
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Success",
            stderr=""
        )

        full_code = {
            'project_json_src': '{"op": "AddCustom"}',
            'host_tiling_src': '// tiling code',
            'host_operator_src': '// operator code',
            'kernel_src': '// kernel code',
            'python_bind_src': '// python binding',
            'model_src': '// model code'
        }

        result = self.evaluator.compile(full_code, "add")

        # Verify result
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])

        # Verify context was populated
        self.assertEqual(self.evaluator.context, full_code)
        self.assertEqual(self.evaluator.current_op, "add_custom")

    @patch('subprocess.run')
    def test_compile_msopgen_failure(self, mock_subprocess):
        """Test compilation failure at msopgen stage."""
        # Mock msopgen failure
        mock_subprocess.side_effect = [
            MagicMock(  # First call: msopgen (fails)
                returncode=1,
                stdout="msopgen error",
                stderr="Failed to generate project"
            )
        ]

        full_code = {
            'project_json_src': '{"op": "AddCustom"}',
            'host_tiling_src': '// tiling code',
            'host_operator_src': '// operator code',
            'kernel_src': '// kernel code',
            'python_bind_src': '// python binding',
            'model_src': '// model code'
        }

        # Need to mock file operations
        with patch('builtins.open', mock_open()):
            with patch('os.path.exists', return_value=False):
                with patch('os.chdir'):
                    result = self.evaluator.compile(full_code, "add")

        # Verify failure was caught
        self.assertFalse(result["success"])
        self.assertIn("msopgen failed", result["error"])

    def test_compile_missing_code_components(self):
        """Test compilation with missing code components."""
        full_code = {
            'project_json_src': '{"op": "AddCustom"}',
            # Missing other components
        }

        with patch('builtins.open', mock_open()):
            with patch('os.path.exists', return_value=False):
                with patch('os.chdir'):
                    with patch('subprocess.run') as mock_subprocess:
                        mock_subprocess.return_value = MagicMock(
                            returncode=0, stdout="", stderr=""
                        )
                        result = self.evaluator.compile(full_code, "add")

        # Should not fail - empty strings will be written
        # Actual compilation would fail later at build stage

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.chdir')
    def test_deploy_success(self, mock_chdir, mock_file, mock_subprocess):
        """Test successful deployment."""
        # Mock subprocess calls (deployment and pybind)
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Deployment successful",
            stderr=""
        )

        # Setup evaluator state from previous compile
        self.evaluator.context = {
            'model_src': 'import torch\nclass ModelNew: pass'
        }

        with patch('os.environ', {}):
            result = self.evaluator.deploy("add")

        # Verify result
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])

    @patch('subprocess.run')
    def test_deploy_failure(self, mock_subprocess):
        """Test deployment failure."""
        # Mock deployment failure
        mock_subprocess.side_effect = Exception("Deployment error")

        self.evaluator.context = {'model_src': ''}

        with patch('os.chdir'):
            result = self.evaluator.deploy("add")

        # Verify failure
        self.assertFalse(result["success"])
        self.assertIn("error", result["error"].lower())

    def test_cleanup(self):
        """Test cleanup method."""
        self.evaluator.context = {'test': 'data'}

        # Mock torch_npu
        with patch('evotoolkit.task.cann_init.evaluator.torch_npu') as mock_torch_npu:
            self.evaluator.cleanup()

        # Verify context was cleared
        self.assertEqual(self.evaluator.context, {})

        # Verify cleanup methods were called
        mock_torch_npu.npu.empty_cache.assert_called_once()
        mock_torch_npu.npu.synchronize.assert_called_once()


class TestEvaluatorHelperFunctions(unittest.TestCase):
    """Test helper functions in the evaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.evaluator = AscendCEvaluator(project_path=self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_underscore_to_pascalcase_edge_cases(self):
        """Test edge cases for underscore to PascalCase conversion."""
        test_cases = [
            ("", ""),
            ("a", "A"),
            ("abc", "Abc"),
            ("a_b_c", "ABC"),
            ("_test", "Test"),
            ("test_", "Test"),
            ("__test__", "Test"),
        ]

        for input_str, expected in test_cases:
            with self.subTest(input=input_str):
                result = self.evaluator._underscore_to_pascalcase(input_str)
                self.assertEqual(result, expected)


def run_unit_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAscendCEvaluatorUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluatorHelperFunctions))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_unit_tests()
    sys.exit(0 if success else 1)
