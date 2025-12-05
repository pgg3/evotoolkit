# AscendCEvaluator Tests

This directory contains tests for the `AscendCEvaluator` class.

## Test Files

### 1. `test_evaluator_unit.py`
Unit tests that verify the basic functionality of each method in the evaluator.

**Features:**
- Tests initialization
- Tests helper functions (e.g., `_underscore_to_pascalcase`)
- Mocks external dependencies (subprocess, file I/O)
- Does NOT require NPU hardware
- Fast execution (< 1 second)

**Run:**
```bash
cd /root/Huawei_CANN/evotoolkit/tests
python test_evaluator_unit.py
```

### 2. `test_evaluator_integration.py`
Integration test that performs end-to-end testing with a real Add operator.

**Features:**
- Tests complete compilation pipeline
- Tests deployment
- Tests correctness verification
- Tests performance measurement
- Uses actual Add operator code from MultiKernelBench

**Requirements:**
- Ascend NPU hardware (Ascend910B or compatible)
- CANN toolkit installed and configured
- `msopgen` available in PATH
- `torch_npu` installed
- Build tools (gcc, cmake, etc.)

**Run:**
```bash
cd /root/Huawei_CANN/evotoolkit/tests
python test_evaluator_integration.py
```

## Quick Start

### Run Unit Tests (No NPU Required)
```bash
# From evotoolkit directory
python tests/test_evaluator_unit.py
```

Expected output:
```
test_compile_missing_code_components (__main__.TestAscendCEvaluatorUnit) ... ok
test_compile_msopgen_failure (__main__.TestAscendCEvaluatorUnit) ... ok
test_compile_success (__main__.TestAscendCEvaluatorUnit) ... ok
...
Ran 10 tests in 0.XXXs

OK
```

### Run Integration Tests (Requires NPU)
```bash
# Ensure CANN environment is set up
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # Or your CANN path

# Run integration test
python tests/test_evaluator_integration.py
```

Expected output:
```
======================================================================
ASCEND C EVALUATOR - INTEGRATION TEST
======================================================================

torch_npu version: X.X.X
NPU available: True
...

======================================================================
TEST SUMMARY
======================================================================
  Compile: PASS
  Deploy: PASS
  Correctness: PASS
  Performance: PASS

Overall: PASS
```

## Troubleshooting

### Unit Tests

**Issue:** `ImportError: No module named 'evotoolkit'`
- **Solution:** Make sure you're in the tests directory and the parent path is correct

**Issue:** Tests fail with mock errors
- **Solution:** Install mock library: `pip install mock`

### Integration Tests

**Issue:** `torch_npu not available`
- **Solution:** Install torch_npu: `pip install torch_npu`

**Issue:** `msopgen: command not found`
- **Solution:** Source CANN environment: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

**Issue:** Compilation fails with build errors
- **Solution:** Check that CANN toolkit is properly installed and environment variables are set

**Issue:** Deployment fails
- **Solution:** Check permissions and ensure custom operator path is writable

**Issue:** Correctness check fails
- **Solution:** Verify NPU is accessible: `npu-smi info`

## Test Coverage

### Unit Tests Coverage
- ✅ Evaluator initialization
- ✅ Helper functions
- ✅ Compile method (mocked)
- ✅ Deploy method (mocked)
- ✅ Cleanup method
- ✅ Error handling

### Integration Tests Coverage
- ✅ Full compilation pipeline
- ✅ Operator deployment
- ✅ Correctness verification against reference
- ✅ Performance measurement
- ✅ End-to-end workflow

## Adding New Tests

To add a new unit test:
1. Add a test method to `TestAscendCEvaluatorUnit` class
2. Use `self.assertEqual()`, `self.assertTrue()`, etc.
3. Mock external dependencies as needed

To add a new integration test:
1. Create a new test function in `test_evaluator_integration.py`
2. Follow the pattern: setup → execute → verify → cleanup
3. Add the test to `run_integration_test()`

## CI/CD Integration

For CI/CD pipelines:

```bash
# Run unit tests (always)
python tests/test_evaluator_unit.py || exit 1

# Run integration tests (only if NPU available)
if npu-smi info &> /dev/null; then
    python tests/test_evaluator_integration.py || exit 1
fi
```
