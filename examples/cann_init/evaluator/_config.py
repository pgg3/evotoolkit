# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SRC_DIR = Path(__file__).parent / "0_test_task_src"
OUTPUT_DIR = Path(__file__).parent / "output"

PYTHON_REFERENCE = (SRC_DIR / "python_reference.py").read_text()

# Kernel implementation (class and helper code)
KERNEL_IMPL = """constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR output, uint32_t totalLength, uint32_t tileNum)
    {
        // Each core processes blockLength elements
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // Each tile processes tileLength elements
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGm.SetGlobalBuffer((__gm__ float*)output + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(outputGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};"""

# Kernel entry function body (after GET_TILING_DATA)
KERNEL_ENTRY_BODY = """    KernelAdd op;
    op.Init(x, y, output, tilingData.totalLength, tilingData.tileNum);
    op.Process();"""

TILING_FIELDS = [
    {"name": "totalLength", "type": "uint32_t"},
    {"name": "tileNum", "type": "uint32_t"},
]

TILING_FUNC_BODY = """    AddCustomTilingData tiling;

    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalLength *= shape.GetDim(i);
    }

    constexpr uint32_t BLOCK_DIM = 8;
    uint32_t tileNum = BLOCK_DIM;

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(BLOCK_DIM);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;"""

INFER_SHAPE_BODY = """    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;"""

OUTPUT_ALLOC_CODE = "at::Tensor result = at::empty_like(x);"


def ensure_output_dir(subdir: str = "") -> Path:
    path = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_task_data(op_name: str = "add", npu_type: str = "Ascend910B") -> dict:
    return {
        "op_name": op_name,
        "npu_type": npu_type,
        "python_reference": PYTHON_REFERENCE,
    }
