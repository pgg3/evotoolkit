# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import time
import numpy as np
from typing import List, Optional
from evotoolkit.core import Solution
from .island import Island


class ProgramsDatabase:
    """A collection of programs, organized as islands."""
    
    def __init__(
        self,
        num_islands: int = 10,
        solutions_per_prompt: int = 2,
        reset_period: int = 4 * 60 * 60,  # 4 hours in seconds
        cluster_sampling_temperature_init: float = 0.1,
        cluster_sampling_temperature_period: int = 30000,
    ):
        self.num_islands = num_islands
        self.solutions_per_prompt = solutions_per_prompt
        self.reset_period = reset_period
        
        # Initialize empty islands
        self.islands: List[Island] = []
        for _ in range(num_islands):
            island = Island(
                solutions_per_prompt=solutions_per_prompt,
                cluster_sampling_temperature_init=cluster_sampling_temperature_init,
                cluster_sampling_temperature_period=cluster_sampling_temperature_period
            )
            self.islands.append(island)
        
        self.best_scores_per_island: List[float] = [-float('inf')] * num_islands
        self.best_solutions_per_island: List[Optional[Solution]] = [None] * num_islands
        self.last_reset_time: float = time.time()
    
    def register_solution(self, solution: Solution, island_id: Optional[int] = None) -> None:
        """Registers solution in the database."""
        if not solution.evaluation_res or not solution.evaluation_res.valid:
            return
        
        score = solution.evaluation_res.score
        
        if island_id is None:
            # Initial solution - add to all islands
            for i in range(self.num_islands):
                self._register_solution_in_island(solution, i, score)
        else:
            # Register in specific island
            self._register_solution_in_island(solution, island_id, score)
        
        # Check if it's time to reset islands
        if time.time() - self.last_reset_time > self.reset_period:
            self.last_reset_time = time.time()
            self.reset_islands()
    
    def _register_solution_in_island(self, solution: Solution, island_id: int, score: float) -> None:
        """Registers solution in the specified island."""
        self.islands[island_id].register_solution(solution, score)
        
        if score > self.best_scores_per_island[island_id]:
            self.best_solutions_per_island[island_id] = solution
            self.best_scores_per_island[island_id] = score
    
    def get_prompt_solutions(self) -> tuple[List[Solution], int]:
        """Returns solutions from a randomly chosen island for prompt generation."""
        island_id = np.random.randint(self.num_islands)
        solutions = self.islands[island_id].get_prompt_solutions()
        return solutions, island_id
    
    def get_best_solution(self) -> Optional[Solution]:
        """Returns the globally best solution."""
        best_island_id = np.argmax(self.best_scores_per_island)
        return self.best_solutions_per_island[best_island_id]
    
    def get_best_score(self) -> float:
        """Returns the globally best score."""
        return max(self.best_scores_per_island)
    
    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # Sort islands by score with minor noise to break ties
        scores_with_noise = np.array(self.best_scores_per_island) + np.random.randn(self.num_islands) * 1e-6
        indices_sorted_by_score = np.argsort(scores_with_noise)
        
        num_islands_to_reset = self.num_islands // 2
        reset_island_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_island_ids = indices_sorted_by_score[num_islands_to_reset:]
        
        for island_id in reset_island_ids:
            # Reset the island
            self.islands[island_id] = Island(
                solutions_per_prompt=self.solutions_per_prompt,
                cluster_sampling_temperature_init=self.islands[island_id].cluster_sampling_temperature_init,
                cluster_sampling_temperature_period=self.islands[island_id].cluster_sampling_temperature_period
            )
            self.best_scores_per_island[island_id] = -float('inf')
            
            # Add a founder from a good island
            founder_island_id = np.random.choice(keep_island_ids)
            founder_solution = self.best_solutions_per_island[founder_island_id]
            founder_score = self.best_scores_per_island[founder_island_id]
            
            if founder_solution:
                self._register_solution_in_island(founder_solution, island_id, founder_score)
    
    def get_statistics(self) -> dict:
        """Returns database statistics."""
        total_programs = sum(island.num_programs for island in self.islands)
        island_stats = []
        
        for i, island in enumerate(self.islands):
            island_stats.append({
                'island_id': i,
                'num_programs': island.num_programs,
                'num_clusters': len(island.clusters),
                'best_score': self.best_scores_per_island[i]
            })
        
        return {
            'total_programs': total_programs,
            'num_islands': self.num_islands,
            'global_best_score': self.get_best_score(),
            'island_stats': island_stats
        }
    
    def to_dict(self) -> dict:
        """Serialize the database to a dictionary."""
        islands_data = []
        for island in self.islands:
            island_data = {
                'clusters': {},
                'num_programs': island.num_programs,
                'solutions_per_prompt': island.solutions_per_prompt,
                'cluster_sampling_temperature_init': island.cluster_sampling_temperature_init,
                'cluster_sampling_temperature_period': island.cluster_sampling_temperature_period
            }
            
            # Serialize clusters
            for score, cluster in island.clusters.items():
                cluster_data = {
                    'score': cluster.score,
                    'solutions': [],
                    'lengths': cluster.lengths
                }
                
                # Serialize solutions in cluster
                for solution in cluster.solutions:
                    sol_dict = {
                        'sol_string': solution.sol_string,
                        'other_info': solution.other_info,
                        'evaluation_res': None
                    }
                    if solution.evaluation_res:
                        sol_dict['evaluation_res'] = {
                            'valid': solution.evaluation_res.valid,
                            'score': solution.evaluation_res.score,
                            'additional_info': solution.evaluation_res.additional_info
                        }
                    cluster_data['solutions'].append(sol_dict)
                
                island_data['clusters'][str(score)] = cluster_data
            
            islands_data.append(island_data)
        
        return {
            'num_islands': self.num_islands,
            'solutions_per_prompt': self.solutions_per_prompt,
            'reset_period': self.reset_period,
            'islands': islands_data,
            'best_scores_per_island': self.best_scores_per_island,
            'last_reset_time': self.last_reset_time
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProgramsDatabase':
        """Deserialize the database from a dictionary."""
        from evotoolkit.core import Solution, EvaluationResult
        from .island import Island, Cluster
        
        # Create database with config
        database = cls(
            num_islands=data['num_islands'],
            solutions_per_prompt=data['solutions_per_prompt'],
            reset_period=data['reset_period']
        )
        
        database.last_reset_time = data.get('last_reset_time', database.last_reset_time)
        database.best_scores_per_island = data.get('best_scores_per_island', database.best_scores_per_island)
        
        # Restore islands
        for i, island_data in enumerate(data['islands']):
            island = Island(
                solutions_per_prompt=island_data['solutions_per_prompt'],
                cluster_sampling_temperature_init=island_data['cluster_sampling_temperature_init'],
                cluster_sampling_temperature_period=island_data['cluster_sampling_temperature_period']
            )
            island.num_programs = island_data['num_programs']
            
            # Restore clusters
            for score_str, cluster_data in island_data['clusters'].items():
                score = float(score_str)
                
                # Restore solutions in cluster
                solutions = []
                for sol_dict in cluster_data['solutions']:
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
                    solutions.append(solution)
                
                # Create cluster with first solution
                if solutions:
                    cluster = Cluster(score, solutions[0])
                    # Add remaining solutions
                    for solution in solutions[1:]:
                        cluster.register_solution(solution)
                    # Restore lengths
                    cluster.lengths = cluster_data['lengths']
                    island.clusters[score] = cluster
            
            database.islands[i] = island
            
            # Update best solution for this island
            if island.clusters:
                best_score = max(island.clusters.keys())
                database.best_solutions_per_island[i] = island.clusters[best_score].sample_solution()
        
        return database