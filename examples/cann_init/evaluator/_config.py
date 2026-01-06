# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SRC_DIR = Path(__file__).parent / "0_test_task_src"
OUTPUT_DIR = Path(__file__).parent / "output"

PYTHON_REFERENCE = (SRC_DIR / "python_reference.py").read_text()
KERNEL_SRC = (SRC_DIR / "kernel_src.cpp").read_text()

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

INFER_DTYPE_BODY = None


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
