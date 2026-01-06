# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Prompt generation for EvoEngineer CANN Interface"""

import json
from typing import List

from evotoolkit.core import Solution


class PromptMixin:
    """Mixin for prompt generation methods"""

    def _get_tiling_data_class_name(self) -> str:
        """Generate TilingData class name from op_name"""
        op_name = self.task.op_name
        # Convert to PascalCase and add CustomTilingData
        pascal_name = "".join(word.capitalize() for word in op_name.split("_"))
        return f"{pascal_name}CustomTilingData"

    def _get_task_description(self) -> str:
        """Get task description from CANNInitTask"""
        return self.task.get_base_task_description()

    def _format_solution(self, sol: Solution) -> str:
        """Format a solution for display in prompt"""
        if not sol or not sol.sol_string:
            return "(no solution)"

        info = sol.other_info or {}
        runtime = ""
        if sol.evaluation_res and sol.evaluation_res.score is not None:
            runtime = f"Runtime: {-sol.evaluation_res.score:.4f} ms\n"

        return f"""Name: {info.get('name', 'unnamed')}
{runtime}Thought: {info.get('thought', '')}

kernel_src:
```cpp
{sol.sol_string}
```

tiling_fields:
```json
{json.dumps(info.get('tiling_fields', []), indent=2)}
```

tiling_func_body:
```cpp
{info.get('tiling_func_body', '')}
```

infer_shape_body:
```cpp
{info.get('infer_shape_body', '')}
```"""

    def _get_response_format(self) -> str:
        tiling_class_name = self._get_tiling_data_class_name()
        return f"""## RESPONSE FORMAT
name: [descriptive_name]
thought: [optimization rationale]

kernel_src:
```cpp
[Ascend C kernel code - IMPORTANT: All Ascend C APIs must use AscendC:: namespace prefix]
[Example structure:
  #include "kernel_operator.h"

  constexpr int32_t BUFFER_NUM = 2;

  class KernelAdd {{
  public:
      __aicore__ inline KernelAdd() {{}}
      __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, ...) {{
          xGm.SetGlobalBuffer((__gm__ DTYPE*)x);
          // ... setup queues with pipe.InitBuffer
      }}
      __aicore__ inline void Process() {{
          // Process loop with CopyIn, Compute, CopyOut pipeline
      }}
  private:
      AscendC::TPipe pipe;
      AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
      AscendC::GlobalTensor<DTYPE> xGm;
      // ... other members
  }};

  extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                     GM_ADDR workspace, GM_ADDR tiling) {{
      GET_TILING_DATA(tiling_data, tiling);
      KernelAdd op;
      op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileLength);
      op.Process();
  }}
]
[Common APIs that need AscendC:: prefix:
  - AscendC::GlobalTensor<T>
  - AscendC::LocalTensor<T>
  - AscendC::TPipe
  - AscendC::TQue<AscendC::QuePosition::VECIN/VECOUT, N>
  - AscendC::GetBlockNum(), AscendC::GetBlockIdx()
  - AscendC::DataCopy(dst, src, len)
  - AscendC::Add(dst, src1, src2, len)
]
```

tiling_fields:
```json
[{{"name": "field1", "type": "uint32_t"}}, ...]
```

tiling_func_body:
```cpp
[TilingFunc body - this code runs on CPU to calculate tiling parameters]
[Available context: gert::TilingContext* context, and {tiling_class_name} tiling]
[Example API usage:
  auto shape = context->GetInputShape(0)->GetStorageShape();
  uint32_t totalLength = 1;
  for (size_t i = 0; i < shape.GetDimNum(); i++) {{
      totalLength *= shape.GetDim(i);
  }}

  {tiling_class_name} tiling;
  tiling.set_field1(value1);
  tiling.set_field2(value2);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(8);  // Number of AI cores to use

  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;  // Workspace size if needed

  return ge::GRAPH_SUCCESS;
]
```

infer_shape_body:
```cpp
[InferShape body - infer output shape from input shapes]
[Available context: gert::InferShapeContext* context]
[Example API usage:
  const gert::Shape* input_shape = context->GetInputShape(0);
  gert::Shape* output_shape = context->GetOutputShape(0);
  *output_shape = *input_shape;  // For element-wise ops
  return GRAPH_SUCCESS;
]
```"""

    def _get_init_prompt(self, task_desc: str, current_best: Solution, thoughts: List[str]) -> List[dict]:
        thoughts_section = ""
        if thoughts:
            thoughts_section = "\n## OPTIMIZATION INSIGHTS\n" + "\n".join(f"- {t}" for t in thoughts)

        best_section = ""
        if current_best and current_best.sol_string:
            best_section = f"\n## CURRENT BEST\n{self._format_solution(current_best)}"

        prompt = f"""# ASCEND C KERNEL TASK

{task_desc}
{best_section}
{thoughts_section}

## TASK
Implement a high-performance Ascend C kernel for this operator.

{self._get_response_format()}"""

        return [{"role": "user", "content": prompt}]

    def _get_crossover_prompt(
        self, task_desc: str, parents: List[Solution], current_best: Solution, thoughts: List[str]
    ) -> List[dict]:
        thoughts_section = ""
        if thoughts:
            thoughts_section = "\n## OPTIMIZATION INSIGHTS\n" + "\n".join(f"- {t}" for t in thoughts)

        parents_section = "\n## PARENTS TO COMBINE\n"
        for i, p in enumerate(parents, 1):
            parents_section += f"\n### Parent {i}\n{self._format_solution(p)}\n"

        prompt = f"""# ASCEND C KERNEL CROSSOVER

{task_desc}

## CURRENT BEST
{self._format_solution(current_best)}
{parents_section}
{thoughts_section}

## TASK
Combine the best features from both parent implementations.

{self._get_response_format()}"""

        return [{"role": "user", "content": prompt}]

    def _get_mutation_prompt(
        self, task_desc: str, individual: Solution, current_best: Solution, thoughts: List[str]
    ) -> List[dict]:
        thoughts_section = ""
        if thoughts:
            thoughts_section = "\n## OPTIMIZATION INSIGHTS\n" + "\n".join(f"- {t}" for t in thoughts)

        prompt = f"""# ASCEND C KERNEL MUTATION

{task_desc}

## CURRENT BEST
{self._format_solution(current_best)}

## SOURCE TO MUTATE
{self._format_solution(individual)}
{thoughts_section}

## TASK
Apply significant modifications to explore new optimization directions.

{self._get_response_format()}"""

        return [{"role": "user", "content": prompt}]
