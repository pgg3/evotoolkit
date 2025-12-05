#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Integration test for AscendCEvaluator.

This test performs a complete end-to-end test of the evaluator using
the actual Add operator from MultiKernelBench as an example.

NOTE: This test requires:
- Ascend NPU hardware (Ascend910B or compatible)
- CANN toolkit installed
- msopgen, build tools available
- torch_npu installed
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evotoolkit.task.cann_init.evaluator import AscendCEvaluator


# Sample Add operator code from MultiKernelBench
SAMPLE_ADD_OPERATOR = {
    'project_json_src': '''[
    {
        "op": "AddCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ],
        "output_desc": [
            {
                "name": "z",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ]
    }
]''',

    'host_tiling_src': '''#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}''',

    'host_operator_src': '''#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AddCustom);
}''',

    'kernel_src': '''#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdd op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}''',

    'python_bind_src': '''#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor add_custom_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnAddCustom, self, other, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("add_custom", &add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_custom", &add_custom_impl_npu, "x + y");
}''',

    'model_src': '''import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return custom_ops_lib.add_custom(a, b)
'''
}

# Python reference implementation for Add
PYTHON_REFERENCE_ADD = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """Simple model that performs element-wise addition."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors element-wise."""
        return x + y

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    return [x, y]

def get_init_inputs():
    return []  # No special initialization inputs needed
'''


def test_compile():
    """Test the compile method with real Add operator code."""
    print("\n" + "="*70)
    print("TEST 1: Compile")
    print("="*70)

    test_dir = tempfile.mkdtemp(prefix="test_ascendc_eval_")
    print(f"Test directory: {test_dir}")

    try:
        evaluator = AscendCEvaluator(project_path=test_dir)

        print("\nCompiling Add operator...")
        result = evaluator.compile(SAMPLE_ADD_OPERATOR, "add")

        print(f"\nCompile result:")
        print(f"  Success: {result['success']}")
        if result['error']:
            print(f"  Error: {result['error']}")

        return result['success'], evaluator, test_dir

    except Exception as e:
        print(f"\n[ERROR] Exception during compile: {e}")
        import traceback
        traceback.print_exc()
        return False, None, test_dir


def test_deploy(evaluator):
    """Test the deploy method."""
    print("\n" + "="*70)
    print("TEST 2: Deploy")
    print("="*70)

    try:
        print("\nDeploying Add operator...")
        result = evaluator.deploy("add")

        print(f"\nDeploy result:")
        print(f"  Success: {result['success']}")
        if result['error']:
            print(f"  Error: {result['error']}")

        return result['success']

    except Exception as e:
        print(f"\n[ERROR] Exception during deploy: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verify_correctness(evaluator):
    """Test the verify_correctness method."""
    print("\n" + "="*70)
    print("TEST 3: Verify Correctness")
    print("="*70)

    try:
        print("\nVerifying correctness...")
        result = evaluator.verify_correctness(PYTHON_REFERENCE_ADD, "add")

        print(f"\nCorrectness result:")
        print(f"  Pass: {result['pass']}")
        if result.get('error'):
            print(f"  Error: {result['error']}")
        if result.get('max_diff'):
            print(f"  Max diff: {result['max_diff']}")

        return result['pass']

    except Exception as e:
        print(f"\n[ERROR] Exception during correctness check: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_measure_performance(evaluator):
    """Test the measure_performance method."""
    print("\n" + "="*70)
    print("TEST 4: Measure Performance")
    print("="*70)

    try:
        print("\nMeasuring performance...")
        result = evaluator.measure_performance("add")

        print(f"\nPerformance result:")
        if result.get('runtime'):
            print(f"  Mean runtime: {result['runtime']:.3f} ms")
            print(f"  Std dev: {result['std']:.3f} ms")
            print(f"  Min: {result['min']:.3f} ms")
            print(f"  Max: {result['max']:.3f} ms")
            print(f"  Trials: {result['num_trials']}")
            return True
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"\n[ERROR] Exception during performance measurement: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_test():
    """Run complete integration test."""
    print("\n" + "="*70)
    print("ASCEND C EVALUATOR - INTEGRATION TEST")
    print("="*70)

    # Check for NPU availability
    try:
        import torch_npu
        print(f"\ntorch_npu version: {torch_npu.__version__}")
        print(f"NPU available: {torch_npu.npu.is_available()}")
        if torch_npu.npu.is_available():
            print(f"NPU device count: {torch_npu.npu.device_count()}")
            print(f"Current device: {torch_npu.npu.current_device()}")
    except ImportError:
        print("\n[WARNING] torch_npu not available - test will likely fail")

    results = {}

    # Test 1: Compile
    compile_success, evaluator, test_dir = test_compile()
    results['compile'] = compile_success

    if not compile_success:
        print("\n[FAIL] Compilation failed - skipping remaining tests")
        cleanup(test_dir)
        return results

    # Test 2: Deploy
    deploy_success = test_deploy(evaluator)
    results['deploy'] = deploy_success

    if not deploy_success:
        print("\n[FAIL] Deployment failed - skipping remaining tests")
        cleanup(test_dir)
        return results

    # Test 3: Verify Correctness
    correctness_success = test_verify_correctness(evaluator)
    results['correctness'] = correctness_success

    # Test 4: Measure Performance (run even if correctness fails)
    perf_success = test_measure_performance(evaluator)
    results['performance'] = perf_success

    # Cleanup
    cleanup(test_dir)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {test_name.capitalize()}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    print("="*70)

    return results


def cleanup(test_dir):
    """Clean up test directory."""
    import shutil
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"\nWarning: Failed to clean up {test_dir}: {e}")


if __name__ == '__main__':
    results = run_integration_test()
    sys.exit(0 if all(results.values()) else 1)
