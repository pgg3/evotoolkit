# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import concurrent.futures
from typing import List, Type

from evotoolkit.core import BaseRunStateDict, Method, Solution
from evotoolkit.registry import register_algorithm

from .run_config import EvoEngineerConfig
from .run_state_dict import EvoEngineerRunStateDict


@register_algorithm("evoengineer", config=EvoEngineerConfig)
class EvoEngineer(Method):
    def __init__(self, config: EvoEngineerConfig):
        super().__init__(config)
        self.config = config

    def run(self):
        """Main EvoEngineer algorithm execution"""
        self.verbose_title("EvoEngineer ALGORITHM STARTED")

        if "sample" not in self.run_state_dict.usage_history:
            self.run_state_dict.usage_history["sample"] = []

        # Initialize population if starting from scratch (no mandatory seed solution required)
        if self.run_state_dict.generation == 0:
            self._initialize_population()

        # Check if we have enough individuals for selection
        valid_population = self._get_valid_population(self.run_state_dict.population)
        if len(valid_population) < self.config.interface.valid_require:
            self.verbose_info(
                f"The search is terminated since EvoEngineer unable to obtain {self.config.interface.valid_require} feasible algorithms during initialization."
            )
            return

        # Main evolution loop - moved loop control logic here
        while (self.run_state_dict.generation < self.config.max_generations) and (
            self.run_state_dict.tot_sample_nums < self.config.max_sample_nums
        ):
            try:
                self.verbose_info(
                    f"Generation {self.run_state_dict.generation} - Sample {self.run_state_dict.tot_sample_nums + 1} - {self.run_state_dict.tot_sample_nums + self.config.num_samplers} / {self.config.max_sample_nums or 'unlimited'}"
                )

                # Apply offspring operators in parallel for this generation
                self._apply_operators_parallel(
                    self.config.get_offspring_operators(),
                    f"Gen {self.run_state_dict.generation}",
                )

                # Manage population size - keep only the best pop_size individuals
                self._manage_population_size()

                self.run_state_dict.generation += 1
                self._save_run_state_dict()

            except KeyboardInterrupt:
                self.verbose_info("Evolution interrupted by user")
                break
            except Exception as e:
                self.verbose_info(f"Evolution error: {str(e)}")
                continue

        # Mark as done and save final state
        self.run_state_dict.is_done = True
        self._save_run_state_dict()

    def _initialize_population(self):
        """Initialize population using init operators - keep generating until we have enough valid solutions"""
        self.verbose_info("Initializing population...")

        initial_sample_limit = self.config.max_sample_nums  # Reasonable limit

        # Keep generating until we have pop_size valid solutions or hit sample limit
        while self.run_state_dict.tot_sample_nums < initial_sample_limit:
            # Apply init operators in parallel
            self._apply_operators_parallel(self.config.get_init_operators(), "Init")

            valid_count = len(
                self._get_valid_population(self.run_state_dict.population)
            )
            self.verbose_info(f"Valid solutions: {valid_count}/{self.config.pop_size}")

            self._save_run_state_dict()

            if valid_count >= self.config.interface.valid_require:
                break

        valid_population = self._get_valid_population(self.run_state_dict.population)
        if len(valid_population) >= self.config.interface.valid_require:
            self.run_state_dict.generation = 1
            self._save_run_state_dict()
            self.verbose_info(
                f"Initialization completed with {len(valid_population)} valid solutions"
            )
        else:
            self.verbose_info(
                f"Warning: Only {len(valid_population)} valid solutions obtained, need at least {self.config.interface.valid_require}"
            )

    def _get_valid_population(self, population: List[Solution]) -> List[Solution]:
        """Get valid solutions from population"""
        return [
            sol for sol in population if sol.evaluation_res and sol.evaluation_res.valid
        ]

    def _get_best_valid_sol(self, sol_history: List[Solution]) -> Solution | None:
        """Get the best valid solution from sol_history"""
        valid_sols = [
            sol
            for sol in sol_history
            if sol.evaluation_res
            and sol.evaluation_res.valid
            and sol.evaluation_res.score is not None
        ]
        if valid_sols:
            return max(valid_sols, key=lambda x: x.evaluation_res.score)
        return None

    def _register_solution(self, solution: Solution):
        """Register a new solution to both sol_history and population"""
        self.run_state_dict.sol_history.append(solution)
        self.run_state_dict.population.append(solution)
        self.run_state_dict.current_gen_solutions.append(solution)  # 添加到当前代历史
        self.run_state_dict.tot_sample_nums += 1

    def _apply_operators_parallel(self, operators: List, generation_label: str = ""):
        """Apply operators in parallel and register solutions"""
        if not operators:
            return

        # Single executor for both generation and evaluation
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_samplers + self.config.num_evaluators
        ) as executor:
            generate_futures = []
            eval_futures = []

            # Calculate target samples: multiple of num_operators, not exceeding num_samplers
            num_operators = len(operators)

            max_multiplier = self.config.num_samplers // num_operators
            target_samples = (
                max_multiplier * num_operators
            )  # Largest multiple of num_operators <= num_samplers
            samples_per_operator = (
                target_samples // num_operators
            )  # This equals max_multiplier

            # Generate samples: each operator gets exactly samples_per_operator samples
            sample_id = 0
            for operator in operators:
                for _ in range(samples_per_operator):
                    selected_individuals = self._select_individuals_for_operator(
                        operator
                    )
                    future = executor.submit(
                        self._generate_single_solution,
                        operator,
                        selected_individuals,
                        sample_id,
                    )
                    generate_futures.append((operator.name, future))
                    sample_id += 1

            # Process generations as they complete and immediately submit for evaluation
            future_to_operator = {
                future: operator_name for operator_name, future in generate_futures
            }
            for future in concurrent.futures.as_completed(
                [f for _, f in generate_futures]
            ):
                operator_name = future_to_operator[future]
                try:
                    solution, usage = future.result()

                    # Add usage history
                    self.run_state_dict.usage_history["sample"].append(usage)
                    self.run_state_dict.current_gen_usage.append(
                        usage
                    )  # 添加到当前代usage历史

                    # Immediately submit for evaluation without waiting
                    # Support both sol_string (traditional) and other_info (CANN-style) solutions
                    should_evaluate = (
                        solution.sol_string.strip() or
                        (solution.other_info and isinstance(solution.other_info, dict))
                    )

                    if should_evaluate:
                        eval_future = executor.submit(
                            self.config.interface.evaluate, solution
                        )
                        eval_futures.append((eval_future, solution, operator_name))
                    else:
                        self._register_solution(solution)
                        # Log result for empty solution
                        self.verbose_info(
                            f"{operator_name} {generation_label} - Score: None (Invalid)"
                        )

                except Exception as e:
                    self.verbose_info(f"Error generating {operator_name}: {str(e)}")
                    continue

            # Collect evaluation results
            eval_future_to_info = {
                eval_future: (solution, operator_name)
                for eval_future, solution, operator_name in eval_futures
            }
            for eval_future in concurrent.futures.as_completed(
                [ef for ef, _, _ in eval_futures]
            ):
                solution, operator_name = eval_future_to_info[eval_future]
                try:
                    evaluation_res = eval_future.result()
                    solution.evaluation_res = evaluation_res
                    self._register_solution(solution)

                    # Log result
                    score_str = (
                        "None"
                        if not solution.evaluation_res
                        or solution.evaluation_res.score is None
                        else f"{solution.evaluation_res.score}"
                    )
                    valid_str = (
                        "Valid"
                        if solution.evaluation_res and solution.evaluation_res.valid
                        else "Invalid"
                    )
                    self.verbose_info(
                        f"{operator_name} {generation_label} - Score: {score_str} ({valid_str})"
                    )

                except Exception as e:
                    self.verbose_info(f"Error evaluating {operator_name}: {str(e)}")
                    self._register_solution(solution)  # Add with no evaluation result
                    continue

    def _manage_population_size(self):
        """Manage population size - keep only the best pop_size individuals"""
        if len(self.run_state_dict.population) <= self.config.pop_size:
            return

        # Separate valid and invalid solutions
        valid_solutions = self._get_valid_population(self.run_state_dict.population)
        invalid_solutions = [
            sol for sol in self.run_state_dict.population if sol not in valid_solutions
        ]

        # Sort valid solutions by score (descending - higher is better)
        valid_solutions.sort(
            key=lambda x: x.evaluation_res.score
            if x.evaluation_res and x.evaluation_res.score is not None
            else float("-inf"),
            reverse=True,
        )

        # Keep the best valid solutions + some invalid ones if needed
        new_population = []

        # First, add the best valid solutions
        valid_to_keep = min(len(valid_solutions), self.config.pop_size)
        new_population.extend(valid_solutions[:valid_to_keep])

        # If we need more individuals, add some invalid ones (most recent)
        remaining_slots = self.config.pop_size - len(new_population)
        if remaining_slots > 0 and invalid_solutions:
            new_population.extend(
                invalid_solutions[-remaining_slots:]
            )  # Keep most recent invalid ones

        self.run_state_dict.population = new_population

        valid_count = len(self._get_valid_population(new_population))
        self.verbose_info(
            f"Population managed: {len(new_population)} total ({valid_count} valid, {len(new_population) - valid_count} invalid)"
        )

    def _select_individuals_for_operator(self, operator) -> List[Solution]:
        """Select individuals for an operator using rank-based probability selection"""
        import math

        import numpy as np

        if operator.selection_size <= 0:
            return []  # Init operators or invalid selection size

        # Filter valid solutions with finite scores (including NaN check)
        funcs = [
            sol
            for sol in self.run_state_dict.population
            if sol.evaluation_res
            and sol.evaluation_res.valid
            and sol.evaluation_res.score is not None
            and not math.isinf(sol.evaluation_res.score)
            and not math.isnan(sol.evaluation_res.score)
        ]

        if len(funcs) == 0:
            # Fallback to any available solutions
            return (
                self.run_state_dict.population[: operator.selection_size]
                if self.run_state_dict.population
                else []
            )

        # Sort by score (assuming higher is better)
        func = sorted(funcs, key=lambda f: f.evaluation_res.score, reverse=True)

        # Create rank-based probability distribution
        p = [1 / (r + len(func)) for r in range(len(func))]
        p = np.array(p)
        p = p / np.sum(p)

        # Select individuals based on probability
        selected = []
        for _ in range(
            min(operator.selection_size, len(func))
        ):  # Ensure we don't select more than available
            chosen = np.random.choice(func, p=p)
            selected.append(chosen)

        return selected

    def _generate_single_solution(
        self, operator, selected_individuals: List[Solution], sampler_id: int
    ) -> tuple[Solution, dict]:
        """Generate a single solution using an operator"""
        try:
            current_best_sol = self._get_best_sol(self.run_state_dict.population)
            random_3_thought = self._get_n_random_thought(3)
            prompt_content = self.config.interface.get_operator_prompt(
                operator.name, selected_individuals, current_best_sol, random_3_thought
            )
            response, usage = self.config.running_llm.get_response(prompt_content)
            new_sol = self.config.interface.parse_response(response)
            self.verbose_info(
                f"Sampler {sampler_id}: Generated {operator.name} solution"
            )
            return new_sol, usage
        except Exception as e:
            self.verbose_info(
                f"Sampler {sampler_id}: Failed to generate {operator.name} solution - {str(e)}"
            )
            return Solution(""), {}

    def _get_n_random_thought(self, n: int) -> List[str]:
        """Get n random thoughts from solutions in the current population"""
        import random

        # Get all thoughts from current population
        thoughts = []
        for sol in self.run_state_dict.population:
            if sol.other_info and "thought" in sol.other_info:
                thought = sol.other_info["thought"]
                if thought:  # Only add non-empty thoughts
                    thoughts.append(thought)

        # If we don't have enough thoughts, return all available ones
        if len(thoughts) <= n:
            return thoughts

        # Randomly sample n thoughts without replacement
        return random.sample(thoughts, n)

    def _get_run_state_class(self) -> Type[BaseRunStateDict]:
        return EvoEngineerRunStateDict
