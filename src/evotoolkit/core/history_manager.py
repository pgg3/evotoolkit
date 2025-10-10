# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import os
import json
from typing import List, Optional, Dict, Any
from evotoolkit.core import Solution


class HistoryManager:
    """管理进化算法运行历史的保存和加载"""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.history_dir = os.path.join(output_path, "history")
        self.summary_dir = os.path.join(output_path, "summary")

        # 确保目录存在
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

    # ========== By-Generation Methods ==========

    def save_generation_history(
        self,
        generation: int,
        solutions: List[Solution],
        usage: List[Dict],
        statistics: Optional[Dict] = None
    ) -> None:
        """保存某一代的历史记录"""
        gen_file = os.path.join(self.history_dir, f"gen_{generation}.json")

        # 转换Solution对象为字典
        solutions_json = []
        for sol in solutions:
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
            solutions_json.append(sol_dict)

        data = {
            'generation': generation,
            'solutions': solutions_json,
            'usage': usage,
            'statistics': statistics or {}
        }

        with open(gen_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_generation_history(self, generation: int) -> Optional[Dict]:
        """加载某一代的历史记录"""
        gen_file = os.path.join(self.history_dir, f"gen_{generation}.json")
        if not os.path.exists(gen_file):
            return None

        with open(gen_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_all_generations(self) -> List[int]:
        """获取所有已保存的代数"""
        generations = []
        if not os.path.exists(self.history_dir):
            return generations

        for filename in os.listdir(self.history_dir):
            if filename.startswith('gen_') and filename.endswith('.json'):
                try:
                    gen = int(filename.replace('gen_', '').replace('.json', ''))
                    generations.append(gen)
                except ValueError:
                    continue

        return sorted(generations)

    # ========== By-Batch Methods ==========

    def save_batch_history(
        self,
        batch_id: int,
        sample_range: tuple,
        solutions: List[Solution],
        usage: List[Dict],
        metadata: Optional[Dict] = None
    ) -> None:
        """保存批次历史记录（用于FunSearch等）"""
        batch_file = os.path.join(self.history_dir, f"batch_{batch_id:04d}.json")

        # 转换Solution对象为字典
        solutions_json = []
        for sol in solutions:
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
            solutions_json.append(sol_dict)

        data = {
            'batch_id': batch_id,
            'sample_range': sample_range,
            'solutions': solutions_json,
            'usage': usage,
            'metadata': metadata or {}
        }

        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_batch_history(self, batch_id: int) -> Optional[Dict]:
        """加载批次历史记录"""
        batch_file = os.path.join(self.history_dir, f"batch_{batch_id:04d}.json")
        if not os.path.exists(batch_file):
            return None

        with open(batch_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_all_batches(self) -> List[int]:
        """获取所有已保存的批次"""
        batches = []
        if not os.path.exists(self.history_dir):
            return batches

        for filename in os.listdir(self.history_dir):
            if filename.startswith('batch_') and filename.endswith('.json'):
                try:
                    batch = int(filename.replace('batch_', '').replace('.json', ''))
                    batches.append(batch)
                except ValueError:
                    continue

        return sorted(batches)

    # ========== Summary Methods ==========

    def save_usage_history(self, usage_history: Dict) -> None:
        """保存完整的usage历史"""
        usage_file = os.path.join(self.summary_dir, "usage_history.json")
        with open(usage_file, 'w', encoding='utf-8') as f:
            json.dump(usage_history, f, indent=2, ensure_ascii=False)

    def load_usage_history(self) -> Dict:
        """加载usage历史"""
        usage_file = os.path.join(self.summary_dir, "usage_history.json")
        if not os.path.exists(usage_file):
            return {}

        with open(usage_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_best_per_generation(self, best_solutions: List[Dict]) -> None:
        """保存每代最优解摘要"""
        best_file = os.path.join(self.summary_dir, "best_per_generation.json")
        with open(best_file, 'w', encoding='utf-8') as f:
            json.dump(best_solutions, f, indent=2, ensure_ascii=False)

    def load_best_per_generation(self) -> List[Dict]:
        """加载每代最优解摘要"""
        best_file = os.path.join(self.summary_dir, "best_per_generation.json")
        if not os.path.exists(best_file):
            return []

        with open(best_file, 'r', encoding='utf-8') as f:
            return json.load(f)