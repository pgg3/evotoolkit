# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from evotoolkit.core import BaseRunStateDict, Solution
from typing import List

class EvoEngineerRunStateDict(BaseRunStateDict):
    def __init__(
            self,
            task_info: dict,
            generation: int = 0,
            tot_sample_nums: int = 0,
            sol_history: list[Solution] = None,
            population: list[Solution] = None,
            is_done: bool = False
    ):
        super().__init__(task_info)

        self.generation = generation
        self.tot_sample_nums = tot_sample_nums
        self.is_done = is_done
        self.sol_history = sol_history or []  # Complete history of all solutions (kept in memory)
        self.population = population or []     # Current generation population
        self.usage_history = {}

        # 当前代新增的solution（用于历史保存）
        self.current_gen_solutions: List[Solution] = []
        self.current_gen_usage: List[dict] = []
        
    def to_json(self) -> dict:
        """Convert the run state to JSON-serializable dictionary (only current state, no history)"""
        # Convert current population to dictionaries
        population_json = []
        for sol in self.population:
            sol_dict = {
                'sol_string': sol.sol_string,
                'other_info': sol.other_info,
                'evaluation_res': None
            }
            if sol.evaluation_res:
                sol_dict['evaluation_res'] = {
                    'valid': sol.evaluation_res.valid,
                    'score': sol.evaluation_res.score,
                    'additional_info': sol.evaluation_res.additional_info
                }
            population_json.append(sol_dict)

        # 获取当前最优解
        current_best = None
        if self.sol_history:
            valid_sols = [s for s in self.sol_history if s.evaluation_res and s.evaluation_res.valid]
            if valid_sols:
                best_sol = max(valid_sols, key=lambda x: x.evaluation_res.score)
                current_best = {
                    'score': best_sol.evaluation_res.score,
                    'found_at_generation': self.generation,  # 记录发现时的代数（非生成代数）
                    'sol_string': best_sol.sol_string
                }

        return {
            'task_info': self._serialize_value(self.task_info),
            'generation': self.generation,
            'tot_sample_nums': self.tot_sample_nums,
            'population': population_json,
            'is_done': self.is_done,
            'current_best': current_best,
            'metadata': {
                'history_saved_in': 'history/',
                'last_generation': self.generation - 1 if self.generation > 0 else 0
            }
        }
        
    @classmethod
    def from_json(cls, data: dict) -> 'EvoEngineerRunStateDict':
        """Create instance from JSON data (loads current state only, history loaded separately)"""
        from evotoolkit.core import Solution, EvaluationResult

        # Convert population from dictionaries back to Solution objects
        population = []
        for sol_dict in data.get('population', []):
            evaluation_res = None
            if sol_dict.get('evaluation_res'):
                eval_data = sol_dict['evaluation_res']
                evaluation_res = EvaluationResult(
                    valid=eval_data['valid'],
                    score=eval_data['score'],
                    additional_info=eval_data['additional_info']
                )

            solution = Solution(
                sol_string=sol_dict['sol_string'],
                other_info=sol_dict.get('other_info'),
                evaluation_res=evaluation_res
            )
            population.append(solution)

        instance = cls(
            task_info=cls._deserialize_value(data['task_info']),
            generation=data.get('generation', 0),
            tot_sample_nums=data.get('tot_sample_nums', 0),
            sol_history=[],  # History will be loaded separately if needed
            population=population,
            is_done=data.get('is_done', False),
        )
        return instance

    def save_current_history(self) -> None:
        """保存当前代的历史记录"""
        if not self._history_manager:
            return

        if not self.current_gen_solutions:
            return

        # 计算统计信息
        valid_sols = [s for s in self.current_gen_solutions if s.evaluation_res and s.evaluation_res.valid]
        statistics = {
            'total_solutions': len(self.current_gen_solutions),
            'valid_solutions': len(valid_sols),
            'valid_rate': len(valid_sols) / len(self.current_gen_solutions) if self.current_gen_solutions else 0,
        }

        if valid_sols:
            scores = [s.evaluation_res.score for s in valid_sols]
            statistics['avg_score'] = sum(scores) / len(scores)
            statistics['best_score'] = max(scores)
            statistics['worst_score'] = min(scores)

        # 保存这一代的历史
        self._history_manager.save_generation_history(
            generation=self.generation - 1,  # 保存的是上一代的数据
            solutions=self.current_gen_solutions,
            usage=self.current_gen_usage,
            statistics=statistics
        )

        # 保存usage_history摘要
        self._history_manager.save_usage_history(self.usage_history)

        # 清空当前代缓存
        self.current_gen_solutions = []
        self.current_gen_usage = []