# 架构设计

EvoToolkit 设计注重模块化和可扩展性。

---

## 核心组件

### 1. 任务 (`evotoolkit.task`)
定义优化问题和评估逻辑。

**关键类:**
- `BaseTask`: 所有任务的基类
- `PythonTask`: Python 代码优化任务
- `CudaTask`: CUDA 内核优化任务
- `ScientificRegressionTask`: 科学符号回归任务

**职责:**
- 定义问题空间
- 实现解评估逻辑
- 提供初始解

### 2. 方法 (`evotoolkit.evo_method`)
实现进化算法（EoH、EvoEngineer、FunSearch）。

**关键类:**
- `BaseMethod`: 所有算法的基类
- `EvoEngineer`: 主要 LLM 驱动进化算法
- `FunSearch`: 函数搜索方法
- `EoH`: 启发式进化方法

**职责:**
- 管理进化过程
- 维护种群
- 选择和变异操作
- 与 LLM 交互

### 3. 接口 (`evotoolkit.core.method_interface`)
任务和方法之间的桥梁，处理特定于算法的适配。

**关键类:**
- `BaseMethodInterface`: 所有接口的基类
- `EvoEngineerPythonInterface`: EvoEngineer 的 Python 任务接口
- `FunSearchPythonInterface`: FunSearch 的 Python 任务接口
- `EoHPythonInterface`: EoH 的 Python 任务接口

**职责:**
- 生成特定于算法的 LLM 提示
- 解析 LLM 响应
- 实现任务特定的算子
- 协调评估

### 4. 注册表 (`evotoolkit.registry`)
任务和算法的自动发现和注册。

**功能:**
- 自动注册任务和方法
- 提供查找和枚举
- 支持插件扩展

---

## 设计模式

### 工厂模式
`evotoolkit.solve()` 创建算法实例：

```python
def solve(interface, **kwargs):
    # 根据接口类型自动选择算法
    method_class = registry.get_method_for_interface(interface)
    config = create_config(interface, **kwargs)
    algorithm = method_class(config)
    return algorithm.run()
```

### 策略模式
接口提供特定于算法的策略：

```python
class BaseMethodInterface:
    def generate_prompt(self, generation, population):
        # 每个接口实现自己的提示策略
        pass

    def mutate(self, solution):
        # 每个接口实现自己的变异策略
        pass
```

### 模板方法模式
基类定义工作流，子类自定义：

```python
class BaseMethod:
    def run(self):
        # 模板方法定义算法骨架
        self.initialize()
        for gen in range(self.max_generations):
            self.evolve_generation()
        return self.get_best_solution()

    def evolve_generation(self):
        # 由子类实现
        raise NotImplementedError
```

---

## 模块组织

```
evotool/
├── __init__.py              # 高级 API
├── core/                    # 基类和抽象
│   ├── base_task.py        # Task 基类
│   ├── solution.py         # Solution 类
│   ├── base_method.py      # Method 基类
│   ├── base_config.py      # Config 基类
│   └── method_interface/   # Interface 基类
├── evo_method/             # 算法实现
│   ├── eoh/               # EoH 实现
│   ├── evoengineer/       # EvoEngineer 实现
│   └── funsearch/         # FunSearch 实现
├── task/                   # 任务实现
│   ├── python_task/       # Python 任务
│   │   ├── __init__.py
│   │   ├── scientific_regression/
│   │   └── method_interface/
│   └── cuda_engineering/  # CUDA 任务
├── tools/                  # 工具
│   ├── llm/               # LLM API 客户端
│   └── data/              # 数据集管理
└── registry.py            # 组件注册
```

---

## 数据流

### 1. 初始化流程

```
用户 → evotoolkit.solve()
    ↓
创建 Config (interface, llm_api, params)
    ↓
创建 Method 实例 (EvoEngineer/FunSearch/EoH)
    ↓
初始化种群
```

### 2. 进化循环

```
对于每一代:
    ↓
Interface.generate_prompt() → 创建 LLM 提示
    ↓
LLM API → 生成新代码
    ↓
Interface.parse_llm_response() → 提取解
    ↓
Task.evaluate() → 评估适应度
    ↓
Method.select() → 选择下一代
```

### 3. 结果返回

```
Method.get_best_solution()
    ↓
保存结果到 output_path
    ↓
返回最佳解给用户
```

---

## 扩展点

### 添加新任务

1. 继承 `BaseTask` 或 `PythonTask`/`CudaTask`
2. 实现 `evaluate()` 方法
3. （可选）在注册表中注册

```python
from evotoolkit.core import BaseTask

class MyNewTask(BaseTask):
    def evaluate(self, solution):
        # 您的评估逻辑
        return fitness
```

### 添加新算法

1. 继承 `BaseMethod`
2. 实现 `run()` 和相关方法
3. 创建对应的 `Config` 类
4. 在注册表中注册

```python
from evotoolkit.core import BaseMethod

class MyNewAlgorithm(BaseMethod):
    def run(self):
        # 您的算法逻辑
        pass
```

### 添加新接口

1. 继承 `BaseMethodInterface`
2. 实现提示生成和响应解析
3. （可选）实现自定义算子

```python
from evotoolkit.core.method_interface import BaseMethodInterface

class MyNewInterface(BaseMethodInterface):
    def generate_prompt(self, generation, population):
        # 提示生成
        pass

    def parse_llm_response(self, response):
        # 响应解析
        pass
```

---

## 性能考虑

### LLM 调用优化
- 批处理请求
- 缓存常见响应
- 并行评估

### 内存管理
- 流式处理大型种群
- 及时清理临时结果
- 使用生成器而非列表

### 并行化
- 支持多进程评估
- GPU 加速（CUDA 任务）
- 分布式进化（未来）

---

## 安全性

### 代码执行
- 沙盒环境（计划中）
- 执行超时
- 资源限制

### API 密钥
- 环境变量存储
- 不在日志中记录
- 安全传输

---

## 测试架构

```
tests/
├── unit/              # 单元测试
│   ├── test_tasks.py
│   ├── test_methods.py
│   └── test_interfaces.py
├── integration/       # 集成测试
│   └── test_workflows.py
└── fixtures/          # 测试数据
    └── test_data/
```

---

## 未来扩展

### 计划功能
- 分布式进化
- 更多内置任务
- Web UI
- 实时监控

### API 稳定性
- 核心 API 保持向后兼容
- 实验性功能标记为 `@experimental`
- 弃用策略：至少保留两个主要版本

---

详细的实现指南，请参阅源代码文档。
