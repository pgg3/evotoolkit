# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import tempfile
from typing import Any, Dict, Optional

from evotoolkit.core import BaseTask, EvaluationResult, Solution

from .evaluator import AscendCEvaluator
from .utils.templates import AscendCTemplateGenerator
from .signature_parser import OperatorSignatureParser
from .data_structures import CompileResult, CANNSolutionConfig


class CANNInitTask(BaseTask):
    def __init__(
        self,
        data: Dict[str, Any],
        project_path: Optional[str] = None,
        fake_mode: bool = False,
        verbose: bool = True,
    ):
        self.default_project_path = project_path
        self.fake_mode = fake_mode
        self.verbose = verbose
        self._parser = None
        self._template_gen = None
        super().__init__(data)

    def _process_data(self, data: Dict[str, Any]):
        self.op_name = data["op_name"]
        self.python_reference = data["python_reference"]
        self.npu_type = data.get("npu_type", "Ascend910B2")
        self.cann_version = data.get("cann_version", "8.0")

        self._parser = OperatorSignatureParser()
        self.signature = self._parser.parse(self.python_reference, self.op_name)
        self._template_gen = AscendCTemplateGenerator(self.signature)

        self.task_info = {
            "op_name": self.op_name,
            "python_reference": self.python_reference,
            "npu_type": self.npu_type,
            "cann_version": self.cann_version,
            "signature": self.signature,
        }

    def get_task_type(self) -> str:
        return "CANNInit"

    def get_base_task_description(self) -> str:
        """Get the base task description for prompt generation.

        Returns the basic role and device info. This is the abstract method
        required by BaseTask. For full task description including signature
        and component specification, use get_task_description() instead.
        """
        return self._get_base_description()

    def get_task_description(self) -> str:
        """完整的任务描述，包含所有通用信息。

        包括：
        - 角色定义 + 设备信息
        - Python Reference
        - Operator Signature（输入输出参数）
        - Component Specification（6 组件的定义和模板）

        其他方法（如 FunSearch, EvoLang）可以直接调用此方法获取完整任务描述，
        然后添加各自特有的内容（如 solutions 展示、输出格式）。
        """
        return f"""{self._get_base_description()}

{self._get_signature_summary()}

{self._get_component_specification()}"""

    def _get_base_description(self) -> str:
        """内部方法：基础描述（角色 + 设备 + Reference）"""
        return f"""You are an Ascend C operator development expert.
Your task is to implement the kernel code for the "{self.op_name}" operator.
Target device: {self.npu_type} NPU with CANN {self.cann_version}.

Python Reference:
```python
{self.python_reference}
```"""

    def _get_signature_summary(self) -> str:
        """内部方法：签名摘要"""
        lines = ["## Operator Signature", ""]

        # Inputs
        lines.append("**Inputs (forward parameters):**")
        for inp in self.signature["inputs"]:
            dtype = inp["dtype"]
            tensor_info = "tensor" if inp.get("is_tensor", True) else "scalar"
            lines.append(f"- `{inp['name']}`: {dtype} {tensor_info}")

        # Outputs
        lines.append("")
        lines.append("**Outputs:**")
        for out in self.signature["outputs"]:
            dtype = out["dtype"]
            lines.append(f"- `{out['name']}`: {dtype} tensor")

        # Init params (if any)
        if self.signature.get("init_params"):
            lines.append("")
            lines.append("**Init Parameters (__init__ arguments):**")
            for param in self.signature["init_params"]:
                dtype = param["dtype"]
                default_str = f" = {param['default']}" if "default" in param else ""
                lines.append(f"- `{param['name']}`: {dtype}{default_str}")

        return "\n".join(lines)

    def _get_component_specification(self) -> str:
        """内部方法：完整代码架构说明和组件标注"""
        # 准备变量
        op_name = self.op_name
        op_name_snake = op_name.replace("-", "_").lower()
        op_name_pascal = "".join(word.capitalize() for word in op_name_snake.split("_"))
        # 实际模板使用 _custom 后缀
        op_custom = f"{op_name_snake}_custom"
        op_custom_capital = "".join(word.capitalize() for word in op_custom.split("_"))

        # 输入输出参数
        tensor_inputs = [inp for inp in self.signature["inputs"] if inp.get("is_tensor", True)]
        gm_params = [inp["name"] for inp in tensor_inputs]
        gm_params.extend([out["name"] for out in self.signature["outputs"]])
        gm_signature = ", ".join(f"GM_ADDR {p}" for p in gm_params)
        gm_args = ", ".join(gm_params)

        # 第一个输入名
        first_input = tensor_inputs[0]["name"] if tensor_inputs else "x"

        # Python binding 函数签名
        pybind_params = ", ".join(f"const at::Tensor& {inp['name']}" for inp in tensor_inputs)

        spec = f"""## Code Architecture

Ascend C operator requires 4 source files. You need to provide **6 components** that will be assembled into these files:

| File | Description | Your Components |
|------|-------------|-----------------|
| **kernel_src** | Device kernel running on NPU | KERNEL_IMPL, KERNEL_ENTRY_BODY |
| **host_tiling_src** | TilingData structure definition | TILING_FIELDS |
| **host_operator_src** | Host-side TilingFunc and InferShape | TILING_FUNC_BODY, INFER_SHAPE_BODY |
| **python_bind_src** | PyTorch Python binding | OUTPUT_ALLOC_CODE |

---

## File Templates (Structure)

### 1. kernel_src
```cpp
#include "kernel_operator.h"
// [KERNEL_INCLUDES - optional]

[KERNEL_IMPL]  // Your Kernel class goes here

extern "C" __global__ __aicore__ void {op_custom}(
    {gm_signature}, GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tilingData, tiling);
    [KERNEL_ENTRY_BODY]  // Your entry code goes here
}}
```

### 2. host_tiling_src
```cpp
#include "register/tilingdata_base.h"
// [TILING_INCLUDES - optional]

BEGIN_TILING_DATA_DEF({op_custom_capital}TilingData)
    [TILING_FIELDS]  // Your fields go here
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({op_custom_capital}, {op_custom_capital}TilingData)
```

### 3. host_operator_src
```cpp
#include "{op_custom}_tiling.h"
#include "register/op_def_registry.h"
// [TILING_FUNC_INCLUDES - optional]

namespace optiling {{
static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
    [TILING_FUNC_BODY]  // Your tiling logic goes here
}}

static ge::graphStatus InferShape(gert::InferShapeContext* context) {{
    [INFER_SHAPE_BODY]  // Your shape inference goes here
}}
}}
```

### 4. python_bind_src
```cpp
#include <torch/extension.h>
#include "aclnn_{op_custom}.h"

at::Tensor {op_custom}_impl_npu({pybind_params}) {{
    [OUTPUT_ALLOC_CODE]  // Must define 'result'
    EXEC_NPU_CMD(aclnn{op_custom_capital}, {gm_args}, result);
    return result;
}}
```

---

## Complete Example: Add Operator (x + y → z)

Below is a complete working example showing all 6 components for element-wise addition.

### KERNEL_IMPL
```cpp
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {{
public:
    __aicore__ inline KernelAdd() {{}}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum) {{
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)z + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }}

    __aicore__ inline void Process() {{
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {{
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress) {{
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }}

    __aicore__ inline void Compute(int32_t progress) {{
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }}

    __aicore__ inline void CopyOut(int32_t progress) {{
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    uint32_t blockLength, tileNum, tileLength;
}};
```

### KERNEL_ENTRY_BODY
```cpp
KernelAdd op;
op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
op.Process();
```

### TILING_FIELDS
```
uint32_t totalLength
uint32_t tileNum
```

Format: `TYPE NAME` or `TYPE NAME[SIZE]` or `struct TYPE NAME`

### TILING_FUNC_BODY
```cpp
AddCustomTilingData tiling;

auto shape = context->GetInputShape(0)->GetStorageShape();
uint32_t totalLength = 1;
for (size_t i = 0; i < shape.GetDimNum(); i++) {{
    totalLength *= shape.GetDim(i);
}}

constexpr uint32_t BLOCK_DIM = 8;
tiling.set_totalLength(totalLength);
tiling.set_tileNum(BLOCK_DIM);

tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
context->SetBlockDim(BLOCK_DIM);

size_t* currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = 0;

return ge::GRAPH_SUCCESS;
```

### INFER_SHAPE_BODY
```cpp
const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return ge::GRAPH_SUCCESS;
```

### OUTPUT_ALLOC_CODE
```cpp
at::Tensor result = at::empty_like({first_input});
```

---

## Optional Components

- **KERNEL_INCLUDES**: Extra kernel headers (e.g., `lib/matmul_intf.h`)
- **TILING_INCLUDES**: Extra tiling headers (e.g., `tiling/platform/platform_ascendc.h` for `TCubeTiling`)
- **TILING_FUNC_INCLUDES**: Extra TilingFunc headers
"""
        return spec

    def format_solution_components(self, solution: "Solution") -> str:
        """格式化 solution 的 6 个组件为可读的字符串"""
        if not solution.other_info:
            return "(empty solution)"

        info = solution.other_info
        parts = []

        if info.get("kernel_impl"):
            parts.append(f"### KERNEL_IMPL\n```cpp\n{info['kernel_impl']}\n```")

        if info.get("kernel_entry_body"):
            parts.append(f"### KERNEL_ENTRY_BODY\n```cpp\n{info['kernel_entry_body']}\n```")

        if info.get("tiling_fields"):
            fields_str = self._format_tiling_fields(info["tiling_fields"])
            parts.append(f"### TILING_FIELDS\n```\n{fields_str}\n```")

        if info.get("tiling_includes"):
            includes_str = "\n".join(info["tiling_includes"])
            parts.append(f"### TILING_INCLUDES\n```\n{includes_str}\n```")

        if info.get("tiling_func_body"):
            parts.append(f"### TILING_FUNC_BODY\n```cpp\n{info['tiling_func_body']}\n```")

        if info.get("infer_shape_body"):
            parts.append(f"### INFER_SHAPE_BODY\n```cpp\n{info['infer_shape_body']}\n```")

        if info.get("output_alloc_code"):
            parts.append(f"### OUTPUT_ALLOC_CODE\n```cpp\n{info['output_alloc_code']}\n```")

        return "\n\n".join(parts) if parts else "(no components)"

    def _format_tiling_fields(self, fields: list) -> str:
        """将 tiling_fields 列表格式化为文本格式"""
        lines = []
        for field in fields:
            if field.get("is_struct"):
                # struct TYPE NAME
                lines.append(f"struct {field['type']} {field['name']}")
            elif field.get("size"):
                # TYPE NAME[SIZE]
                lines.append(f"{field['type']} {field['name']}[{field['size']}]")
            else:
                # TYPE NAME
                lines.append(f"{field['type']} {field['name']}")
        return "\n".join(lines)

    def _make_result(
        self,
        valid: bool,
        stage: str,
        score: Optional[float] = None,
        error: Optional[str] = None,
        **extra,
    ) -> EvaluationResult:
        """辅助方法：构造 EvaluationResult"""
        info = {"stage": stage}
        if error:
            info["error"] = error
        info.update(extra)
        return EvaluationResult(valid=valid, score=score, additional_info=info)

    def _run_verify_and_perf(
        self,
        evaluator: AscendCEvaluator,
        config: CANNSolutionConfig,
        kernel_src: str,
        project_path: str,
        extra_info: Optional[Dict] = None,
    ) -> EvaluationResult:
        """执行正确性验证和性能测量的公共逻辑"""
        base_info = {"kernel_src": kernel_src, "project_path": project_path}
        if extra_info:
            base_info.update(extra_info)

        if not config.skip_correctness:
            verify_result = evaluator.verify_correctness(self.python_reference, self.op_name)
            if not verify_result["pass"]:
                return self._make_result(
                    valid=False,
                    stage="correctness",
                    error=verify_result["error"],
                    python_output=verify_result.get("python_output"),
                    ascend_output=verify_result.get("ascend_output"),
                    max_diff=verify_result.get("max_diff"),
                    **base_info,
                )

        if not config.skip_performance:
            perf_result = evaluator.measure_performance(self.op_name, python_reference=self.python_reference)
            runtime = perf_result.get("runtime")

            if runtime is None:
                return self._make_result(
                    valid=False,
                    stage="performance",
                    error=perf_result.get("error", "Performance measurement failed"),
                    **base_info,
                )

            return self._make_result(
                valid=True,
                stage="success",
                score=-runtime,
                runtime=runtime,
                runtime_std=perf_result.get("std"),
                **base_info,
            )

        return self._make_result(valid=True, stage="correctness_only", **base_info)

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:  # noqa: ARG002
        return self._make_result(
            valid=False,
            stage="validation",
            error="CANNInitTask requires evaluate_solution() with other_info containing tiling_fields, tiling_func_body, infer_shape_body",
        )

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        config = CANNSolutionConfig.from_dict(solution.other_info)
        kernel_src = solution.sol_string

        try:
            project_path = config.project_path or self.default_project_path
            if project_path is None:
                project_path = tempfile.mkdtemp(prefix=f"cann_{self.op_name}_")

            # 从已保存结果加载
            if config.load_from:
                evaluator = AscendCEvaluator(project_path=project_path, device=self.npu_type, verbose=self.verbose)
                return self._evaluate_from_loaded(evaluator, config)

            # build_only 模式
            if config.build_only:
                return self._handle_build_only(config, project_path, kernel_src)

            # 验证必要字段 (6 个组件都必须提供)
            required_fields = [
                config.tiling_fields,
                config.tiling_func_body,
                config.infer_shape_body,
                config.output_alloc_code,
                config.kernel_impl,
                config.kernel_entry_body,
            ]
            if not all(required_fields):
                return self._make_result(
                    valid=False,
                    stage="validation",
                    error="Missing required fields: tiling_fields, tiling_func_body, infer_shape_body, output_alloc_code, kernel_impl, kernel_entry_body",
                    kernel_src=kernel_src,
                )

            # 生成完整代码
            full_code = self._template_gen.generate(
                kernel_impl=config.kernel_impl,
                kernel_entry_body=config.kernel_entry_body,
                tiling_fields=config.tiling_fields,
                tiling_func_body=config.tiling_func_body,
                infer_shape_body=config.infer_shape_body,
                project_path=project_path,
                output_alloc_code=config.output_alloc_code,
                tiling_func_includes=config.tiling_func_includes,
                tiling_includes=config.tiling_includes,
                kernel_includes=config.kernel_includes,
            )

            # fake_mode: 仅写入文件
            if self.fake_mode:
                return self._handle_fake_mode(full_code, project_path, kernel_src)

            # setup_only 模式
            if config.setup_only:
                return self._handle_setup_only(full_code, project_path, kernel_src)

            # 完整编译流程
            evaluator = AscendCEvaluator(project_path=project_path, device=self.npu_type, verbose=self.verbose)
            compile_result = evaluator.compile(full_code, self.op_name, project_path=project_path, kernel_src=kernel_src)

            if not compile_result.success:
                return self._make_result(
                    valid=False,
                    stage="compile",
                    error=compile_result.error,
                    kernel_src=kernel_src,
                    project_path=project_path,
                )

            if config.save_compile_to:
                compile_result.save(config.save_compile_to)

            if config.compile_only:
                return self._make_result(
                    valid=True,
                    stage="compile_only",
                    project_path=project_path,
                    kernel_src=kernel_src,
                    compile_result=compile_result,
                )

            return self._run_verify_and_perf(evaluator, config, kernel_src, project_path)

        except Exception as e:
            return self._make_result(
                valid=False,
                stage="exception",
                error=str(e),
                kernel_src=kernel_src,
            )

    def _handle_build_only(
        self, config: CANNSolutionConfig, project_path: str, _kernel_src: str
    ) -> EvaluationResult:
        """处理 build_only 模式"""
        from .utils.backend import ascend_build

        full_code = self._load_full_code(project_path)
        if full_code is None:
            return self._make_result(
                valid=False,
                stage="build",
                error="No full_code found. Run setup_only first.",
                project_path=project_path,
            )

        build_result = ascend_build(op_name=self.op_name, project_path=project_path, full_code=full_code)

        if not build_result["success"]:
            return self._make_result(
                valid=False,
                stage="build",
                error=build_result["error"],
                project_path=project_path,
                kernel_src=full_code.get("kernel_src", ""),
            )

        if config.save_compile_to:
            compile_result = CompileResult(
                success=True,
                project_path=project_path,
                op_name=self.op_name,
                context=build_result.get("context", {}),
                kernel_src=full_code.get("kernel_src", ""),
                full_code=full_code,
            )
            compile_result.save(config.save_compile_to)

        return self._make_result(
            valid=True,
            stage="build_only",
            project_path=project_path,
            kernel_src=full_code.get("kernel_src", ""),
        )

    def _handle_fake_mode(
        self, full_code: Dict, project_path: str, kernel_src: str
    ) -> EvaluationResult:
        """处理 fake_mode: 仅写入文件不编译"""
        from .utils.backend import write_project_files

        write_result = write_project_files(full_code=full_code, op_name=self.op_name, project_path=project_path)

        if not write_result["success"]:
            return self._make_result(
                valid=False,
                stage="write_files",
                error=write_result["error"],
                fake_mode=True,
                project_path=project_path,
                kernel_src=kernel_src,
            )

        return self._make_result(
            valid=True,
            stage="files_written",
            score=1.0,
            fake_mode=True,
            project_path=project_path,
            kernel_src=kernel_src,
            generated_components=list(full_code.keys()),
            files_written=write_result.get("files_written", []),
        )

    def _handle_setup_only(
        self, full_code: Dict, project_path: str, kernel_src: str
    ) -> EvaluationResult:
        """处理 setup_only 模式"""
        from .utils.backend import ascend_setup

        setup_result = ascend_setup(
            full_code=full_code,
            op_name=self.op_name,
            project_path=project_path,
            device=self.npu_type,
        )

        if not setup_result["success"]:
            return self._make_result(
                valid=False,
                stage="setup",
                error=setup_result["error"],
                project_path=project_path,
                kernel_src=kernel_src,
            )

        self._save_full_code(project_path, full_code)

        return self._make_result(
            valid=True,
            stage="setup_only",
            project_path=project_path,
            kernel_src=kernel_src,
            target_directory=setup_result.get("target_directory"),
        )

    def _save_full_code(self, project_path: str, full_code: dict) -> None:
        import json
        import os

        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, "full_code.json"), "w") as f:
            json.dump(full_code, f)

    def _load_full_code(self, project_path: str) -> dict:
        import json
        import os

        path = os.path.join(project_path, "full_code.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _evaluate_from_loaded(
        self, evaluator: AscendCEvaluator, config: CANNSolutionConfig
    ) -> EvaluationResult:
        """从已保存的编译结果加载并继续评估"""
        try:
            compile_result = CompileResult.load(config.load_from)

            if not compile_result.is_loadable():
                return self._make_result(
                    valid=False,
                    stage="load",
                    error="Loaded compile result is not usable",
                    load_from=config.load_from,
                )

            evaluator.project_path = compile_result.project_path

            if not evaluator.rebuild_context(compile_result):
                return self._make_result(
                    valid=False,
                    stage="load",
                    error="Failed to rebuild context from loaded result",
                    load_from=config.load_from,
                )

            return self._run_verify_and_perf(
                evaluator,
                config,
                compile_result.kernel_src,
                compile_result.project_path,
                extra_info={"load_from": config.load_from},
            )

        except Exception as e:
            return self._make_result(
                valid=False,
                stage="load_exception",
                error=str(e),
                load_from=config.load_from,
            )

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("")

    def cleanup(self):
        pass
