# CANNInit 测试

## Evaluator 测试

```bash
cd /root/Huawei_CANN/evotoolkit/examples/cann_init/evaluator
python 1_signature_parser.py
python 2_template_generator.py
python 3_basic_evaluation.py
python 4_parallel_compile.py
```

## Agent 测试

```bash
cd /root/Huawei_CANN/evotoolkit/examples/cann_init/agent

# 1. Phase 0 分析器
python 2_phase0.py hard
# → 输出填入 _config.py PHASE0_CONTEXT

# 2. Pybind Branch (生成 pybind_src.cpp)
python 3_pybind.py hard

# 3. Joint Branch Planning (多轮对话 + 知识检索)
python 4_joint_planning.py hard
# → 输出填入 _config.py JOINT_PLAN_CONTEXT

# 4. Joint Branch Implementation (代码生成)
python 5_joint_impl.py hard
# → 输出: impl_hard/tiling.h, op_host.cpp, op_kernel.cpp

# 5. 评估生成代码 (编译 + 运行 + 验证正确性)
python 7_evaluate.py hard

# 6. E2E 完整测试
python 6_e2e_test.py hard
```

## 单独测试

```bash
# 知识检索单独测试
python 1_knowledge_retrieval.py
```

## 完整流程 (hard case)

```bash
cd /root/Huawei_CANN/evotoolkit/examples/cann_init/agent
python 2_phase0.py hard       # Phase 0: 签名分析
python 3_pybind.py hard       # Pybind: 绑定代码
python 4_joint_planning.py hard  # Planning: 多轮对话
python 5_joint_impl.py hard   # Impl: 代码生成
python 7_evaluate.py hard     # Eval: 编译验证
```

