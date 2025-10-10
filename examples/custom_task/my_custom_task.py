# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
自定义任务完整示例

本示例展示如何在 EvoToolkit 中创建自定义优化任务。包含两个示例任务：
1. MyOptimizationTask - 函数近似任务
2. StringMatchTask - 字符串匹配任务

运行此脚本需要设置环境变量：
- LLM_API_URL: LLM API 的 URL（默认：OpenAI API）
- LLM_API_KEY: LLM API 密钥
"""

import os

import numpy as np

import evotoolkit
from evotoolkit.core import EvaluationResult, Solution
from evotoolkit.task.python_task import EvoEngineerPythonInterface, PythonTask
from evotoolkit.tools.llm import HttpsApi

# ============================================================================
# 示例 1: 函数近似任务
# ============================================================================


class MyOptimizationTask(PythonTask):
    """特定问题优化的自定义任务"""

    def __init__(self, data, target, timeout_seconds=30.0):
        """
        使用特定于问题的数据初始化任务

        Args:
            data: 输入数据（NumPy 数组）
            target: 目标输出值（NumPy 数组）
            timeout_seconds: 代码执行超时时间（秒）
        """
        self.target = target
        super().__init__(data, timeout_seconds)

    def _process_data(self, data):
        """处理输入数据并创建 task_info"""
        self.data = data
        self.task_info = {"data_size": len(data), "description": "函数近似任务"}

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """评估候选代码并返回评估结果"""
        try:
            # 1. 执行代码
            namespace = {"np": np}
            exec(candidate_code, namespace)

            # 2. 检查函数是否存在
            if "my_function" not in namespace:
                return EvaluationResult(
                    valid=False,
                    score=float("-inf"),
                    additional_info={"error": 'Function "my_function" not found'},
                )

            evolved_func = namespace["my_function"]

            # 3. 计算适应度（score 越高越好）
            predictions = np.array([evolved_func(x) for x in self.data])
            mse = np.mean((predictions - self.target) ** 2)
            score = -mse  # 负 MSE，越高越好

            return EvaluationResult(
                valid=True, score=score, additional_info={"mse": mse}
            )

        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": f"Evaluation error: {str(e)}"},
            )

    def get_base_task_description(self) -> str:
        """获取任务描述供 prompt 生成使用"""
        return """你是函数近似专家。

任务：创建一个函数 my_function(x)，使其输出尽可能接近目标值。

要求：
- 定义函数 my_function(x: float) -> float
- 使用数学运算：+, -, *, /, **, np.exp, np.log, np.sin, np.cos 等
- 确保数值稳定性

示例代码：
    import numpy as np

    def my_function(x):
        return np.sin(x)
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """创建初始解"""
        initial_code = '''import numpy as np

def my_function(x):
    """简单线性函数作为基线"""
    return x
'''
        eval_res = self.evaluate_code(initial_code)
        return Solution(sol_string=initial_code, evaluation_res=eval_res)


# ============================================================================
# 示例 2: 字符串匹配任务
# ============================================================================


class StringMatchTask(PythonTask):
    """进化生成目标字符串的函数的任务"""

    def __init__(self, target_string, timeout_seconds=30.0):
        self.target = target_string
        super().__init__(
            data={"target": target_string}, timeout_seconds=timeout_seconds
        )

    def _process_data(self, data):
        """处理输入数据"""
        self.data = data
        self.task_info = {"target": self.target, "target_length": len(self.target)}

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """评估代码"""
        namespace = {}
        try:
            exec(candidate_code, namespace)

            if "generate_string" not in namespace:
                return EvaluationResult(
                    valid=False,
                    score=float("-inf"),
                    additional_info={"error": 'Function "generate_string" not found'},
                )

            generated = namespace["generate_string"]()
            # 编辑距离越小越好，所以用负值作为 score
            distance = self.levenshtein_distance(generated, self.target)
            score = -distance  # 越高越好

            return EvaluationResult(
                valid=True,
                score=score,
                additional_info={"distance": distance, "generated": generated},
            )
        except Exception as e:
            return EvaluationResult(
                valid=False, score=float("-inf"), additional_info={"error": str(e)}
            )

    def levenshtein_distance(self, s1, s2):
        """计算 Levenshtein 编辑距离"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_base_task_description(self) -> str:
        """任务描述"""
        return f"""你是字符串生成专家。

任务：创建一个函数 generate_string()，生成目标字符串 "{self.target}"。

要求：
- 定义函数 generate_string() -> str
- 函数应返回与目标字符串尽可能接近的字符串

示例代码：
    def generate_string():
        return "Hello, World!"
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """创建初始解"""
        initial_code = '''def generate_string():
    """初始简单实现"""
    return ""
'''
        eval_res = self.evaluate_code(initial_code)
        return Solution(sol_string=initial_code, evaluation_res=eval_res)


# ============================================================================
# 主函数：运行示例
# ============================================================================


def run_function_approximation_example():
    """运行函数近似示例"""
    print("=" * 60)
    print("示例 1: 函数近似任务")
    print("=" * 60)

    # 创建任务实例
    data = np.linspace(0, 10, 50)
    target = np.sin(data)  # 目标：近似正弦函数

    task = MyOptimizationTask(data, target)

    # 创建接口
    interface = EvoEngineerPythonInterface(task)

    # 设置 LLM
    llm_api = HttpsApi(
        api_url=os.environ.get(
            "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
        ),
        key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
        model="gpt-4o",
    )

    # 求解
    result = evotoolkit.solve(
        interface=interface,
        output_path="./results/custom_task_func_approx",
        running_llm=llm_api,
        max_generations=5,
    )

    print(f"\n最佳得分: {result.evaluation_res.score:.4f}")
    print(f"最佳 MSE: {result.evaluation_res.additional_info['mse']:.4f}")
    print(f"\n生成的代码:\n{result.sol_string}")


def run_string_match_example():
    """运行字符串匹配示例"""
    print("\n" + "=" * 60)
    print("示例 2: 字符串匹配任务")
    print("=" * 60)

    # 创建任务
    task = StringMatchTask("Hello, EvoToolkit!")

    # 创建接口
    interface = EvoEngineerPythonInterface(task)

    # 设置 LLM
    llm_api = HttpsApi(
        api_url=os.environ.get(
            "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
        ),
        key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
        model="gpt-4o",
    )

    # 求解
    result = evotoolkit.solve(
        interface=interface,
        output_path="./results/custom_task_string_match",
        running_llm=llm_api,
        max_generations=5,
    )

    print(f"\n最佳得分: {result.evaluation_res.score:.4f}")
    print(f"编辑距离: {result.evaluation_res.additional_info['distance']}")
    print(f"生成的字符串: {result.evaluation_res.additional_info['generated']}")
    print(f"\n生成的代码:\n{result.sol_string}")


if __name__ == "__main__":
    # 检查环境变量
    if os.environ.get("LLM_API_KEY") == "your-api-key-here" or not os.environ.get(
        "LLM_API_KEY"
    ):
        print("警告: 请设置环境变量 LLM_API_KEY")
        print("示例: export LLM_API_KEY='your-key-here'")
        exit(1)

    # 运行示例（可以选择运行其中一个或两个）
    run_function_approximation_example()
    # run_string_match_example()  # 取消注释以运行字符串匹配示例
