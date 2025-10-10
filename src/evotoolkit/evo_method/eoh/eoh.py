# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import concurrent.futures
from typing import List, Type

from evotoolkit.core import BaseRunStateDict, Method, Solution
from evotoolkit.registry import register_algorithm

from .run_config import EoHConfig
from .run_state_dict import EoHRunStateDict


@register_algorithm("eoh", config=EoHConfig)
class EoH(Method):
    def __init__(self, config: EoHConfig):
        super().__init__(config)
        self.config = config

    def run(self):
        """Main EoH algorithm execution"""
        self.verbose_title("EOH ALGORITHM STARTED")

        if "sample" not in self.run_state_dict.usage_history:
            self.run_state_dict.usage_history["sample"] = []

        # Initialize with seed solution if sol_history is empty
        if len(self.run_state_dict.sol_history) == 0:
            init_sol = self._get_init_sol()
            if init_sol is None:
                exit()
            self.run_state_dict.sol_history.append(init_sol)
            self.run_state_dict.population.append(init_sol)
            self._save_run_state_dict()
            self.verbose_info(
                f"Initialized with baseline solution (score: {init_sol.evaluation_res.score if init_sol.evaluation_res else 'None'})"
            )

        # Initialize population if starting from scratch
        if self.run_state_dict.generation == 0:
            self._initialize_population()

        # Check if we have enough individuals for selection
        valid_population = self._get_valid_population(self.run_state_dict.population)
        if len(valid_population) < self.config.selection_num:
            self.verbose_info(
                f"The search is terminated since EoH unable to obtain {self.config.selection_num} feasible algorithms during initialization."
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

                # Apply operators in parallel for this generation
                new_solutions = self._apply_operators_parallel()

                # Add new solutions to both sol_history and population
                for sol in new_solutions:
                    self.run_state_dict.sol_history.append(sol)
                    self.run_state_dict.population.append(sol)
                    self.run_state_dict.current_gen_solutions.append(
                        sol
                    )  # 添加到当前代历史
                    self.run_state_dict.tot_sample_nums += 1

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
        """Initialize population using i1 prompt - keep generating until we have enough valid solutions"""
        self.verbose_info("Initializing population...")

        initial_sample_limit = self.config.max_sample_nums  # Reasonable limit

        # Keep generating until we have pop_size valid solutions or hit sample limit
        while (
            len(self._get_valid_population(self.run_state_dict.population))
            < self.config.pop_size
            and self.run_state_dict.tot_sample_nums < initial_sample_limit
        ):
            # Generate and immediately evaluate solutions in parallel
            evaluated_solutions = self._generate_and_evaluate_initial_solutions()

            # Add all solutions to both sol_history and population
            for sol in evaluated_solutions:
                self.run_state_dict.sol_history.append(sol)
                self.run_state_dict.population.append(sol)
                self.run_state_dict.current_gen_solutions.append(
                    sol
                )  # 添加到当前代历史
                self.run_state_dict.tot_sample_nums += 1

                score_str = (
                    "None"
                    if not sol.evaluation_res or sol.evaluation_res.score is None
                    else f"{sol.evaluation_res.score}"
                )
                valid_str = (
                    "Valid"
                    if sol.evaluation_res and sol.evaluation_res.valid
                    else "Invalid"
                )
                self.verbose_info(
                    f"Initial sample {self.run_state_dict.tot_sample_nums} - Score: {score_str} ({valid_str})"
                )

            valid_count = len(
                self._get_valid_population(self.run_state_dict.population)
            )
            self.verbose_info(f"Valid solutions: {valid_count}/{self.config.pop_size}")

            self._save_run_state_dict()

        valid_population = self._get_valid_population(self.run_state_dict.population)
        if len(valid_population) >= self.config.selection_num:
            self.run_state_dict.generation = 1
            self._save_run_state_dict()
            self.verbose_info(
                f"Initialization completed with {len(valid_population)} valid solutions"
            )
        else:
            self.verbose_info(
                f"Warning: Only {len(valid_population)} valid solutions obtained, need at least {self.config.selection_num}"
            )

    def _generate_and_evaluate_initial_solutions(self) -> List[Solution]:
        """Generate and immediately evaluate initial solutions using single executor"""
        evaluated_solutions = []

        # Single executor for both generation and evaluation
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_samplers + self.config.num_evaluators
        ) as executor:
            # Submit all generation tasks
            generate_futures = []
            eval_futures = []

            for sampler_id in range(self.config.num_samplers):
                future = executor.submit(
                    self._generate_single_initial_solution, sampler_id
                )
                generate_futures.append(future)

            # Process generations as they complete and immediately submit for evaluation
            for future in concurrent.futures.as_completed(generate_futures):
                try:
                    new_sol, usage = future.result()
                    self.run_state_dict.usage_history["sample"].append(usage)
                    self.run_state_dict.current_gen_usage.append(
                        usage
                    )  # 添加到当前代usage历史

                    # Immediately submit for evaluation without waiting
                    if new_sol.sol_string.strip():  # Only evaluate non-empty solutions
                        eval_future = executor.submit(
                            self.config.task.evaluate_code, new_sol.sol_string
                        )
                        eval_futures.append((eval_future, new_sol))
                    else:
                        evaluated_solutions.append(new_sol)
                except Exception as e:
                    self.verbose_info(f"Initial solution generation failed: {str(e)}")
                    continue

            # Collect evaluation results
            for eval_future, solution in eval_futures:
                try:
                    evaluation_res = eval_future.result()
                    solution.evaluation_res = evaluation_res
                    evaluated_solutions.append(solution)
                except Exception as e:
                    self.verbose_info(f"Evaluation failed: {str(e)}")
                    evaluated_solutions.append(
                        solution
                    )  # Add with no evaluation result
                    continue

        return evaluated_solutions

    def _generate_single_initial_solution(
        self, sampler_id: int
    ) -> tuple[Solution, dict]:
        """Generate a single initial solution"""
        try:
            prompt_content = self.config.interface.get_prompt_i1()
            response, usage = self.config.running_llm.get_response(prompt_content)
            new_sol = self.config.interface.parse_response(response)
            self.verbose_info(f"Sampler {sampler_id}: Generated initial solution")
            return new_sol, usage
        except Exception as e:
            self.verbose_info(
                f"Sampler {sampler_id}: Failed to generate initial solution - {str(e)}"
            )
            return Solution(""), {}

    def _evaluate_solutions(self, solutions: List[Solution]) -> List[Solution]:
        """Evaluate solutions using multithreading"""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_evaluators
        ) as executor:
            futures = []
            for i, solution in enumerate(solutions):
                if solution.sol_string.strip():  # Only evaluate non-empty solutions
                    future = executor.submit(
                        self.config.task.evaluate_code, solution.sol_string
                    )
                    futures.append((future, i))

            # Collect results
            for future, i in futures:
                try:
                    evaluation_res = future.result()
                    solutions[i].evaluation_res = evaluation_res
                except Exception as e:
                    self.verbose_info(f"Evaluation failed: {str(e)}")
                    continue

        return solutions

    def _get_valid_population(self, population: List[Solution]) -> List[Solution]:
        """Get valid solutions from population"""
        return [
            sol for sol in population if sol.evaluation_res and sol.evaluation_res.valid
        ]

    def _get_best_valid_sol(self, sol_history: List[Solution]) -> Solution:
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
        return sol_history[-1] if sol_history else None

    def _apply_operators_parallel(self) -> List[Solution]:
        """Apply all operators in parallel and return new solutions"""
        new_solutions = []

        # Prepare operator tasks
        operator_tasks = []

        # E1 operator - crossover
        selected_individuals = self._select_individuals(self.config.selection_num)
        if selected_individuals:
            prompt_content = self.config.interface.get_prompt_e1(selected_individuals)
            operator_tasks.append(("E1", prompt_content))

        # E2 operator - guided crossover
        if self.config.use_e2_operator:
            selected_individuals = self._select_individuals(self.config.selection_num)
            if selected_individuals:
                prompt_content = self.config.interface.get_prompt_e2(
                    selected_individuals
                )
                operator_tasks.append(("E2", prompt_content))

        # M1 operator - mutation
        if self.config.use_m1_operator:
            selected_individuals = self._select_individuals(1)
            if selected_individuals:
                selected_individual = selected_individuals[0]
                prompt_content = self.config.interface.get_prompt_m1(
                    selected_individual
                )
                operator_tasks.append(("M1", prompt_content))

        # M2 operator - parameter mutation
        if self.config.use_m2_operator:
            selected_individuals = self._select_individuals(1)
            if selected_individuals:
                selected_individual = selected_individuals[0]
                prompt_content = self.config.interface.get_prompt_m2(
                    selected_individual
                )
                operator_tasks.append(("M2", prompt_content))

        # Execute operators and evaluate in parallel using single executor
        if operator_tasks:
            # Single executor for both generation and evaluation
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.num_samplers + self.config.num_evaluators
            ) as executor:
                generate_futures = []
                eval_futures = []

                # Calculate samples per operator to maintain balanced distribution
                num_operators = len(operator_tasks)

                # Calculate target samples: multiple of num_operators, not exceeding num_samplers
                max_multiplier = self.config.num_samplers // num_operators
                target_samples = (
                    max_multiplier * num_operators
                )  # Largest multiple of num_operators <= num_samplers
                samples_per_operator = (
                    target_samples // num_operators
                )  # This equals max_multiplier

                # Generate samples: each operator gets exactly samples_per_operator samples
                sample_id = 0
                for operator_name, prompt_content in operator_tasks:
                    for _ in range(samples_per_operator):
                        future = executor.submit(
                            self._generate_single_operator_solution,
                            prompt_content,
                            operator_name,
                            sample_id,
                        )
                        generate_futures.append((operator_name, future))
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
                        if solution.sol_string.strip():
                            eval_future = executor.submit(
                                self.config.task.evaluate_code, solution.sol_string
                            )
                            eval_futures.append((eval_future, solution, operator_name))
                        else:
                            new_solutions.append(solution)
                            # Log result for empty solution
                            self.verbose_info(
                                f"{operator_name} Gen {self.run_state_dict.generation} - Score: None (Invalid)"
                            )

                    except Exception as e:
                        self.verbose_info(f"Error generating {operator_name}: {str(e)}")
                        continue

                # Collect evaluation results
                for eval_future, solution, operator_name in eval_futures:
                    try:
                        evaluation_res = eval_future.result()
                        solution.evaluation_res = evaluation_res
                        new_solutions.append(solution)

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
                            f"{operator_name} Gen {self.run_state_dict.generation} - Score: {score_str} ({valid_str})"
                        )

                    except Exception as e:
                        self.verbose_info(f"Error evaluating {operator_name}: {str(e)}")
                        new_solutions.append(solution)  # Add with no evaluation result
                        continue

        return new_solutions

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

    def _select_individuals(self, num_select: int) -> List[Solution]:
        """Select individuals from population using rank-based probability selection"""
        import math

        import numpy as np

        # Handle edge cases
        if num_select <= 0:
            return []

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
                self.run_state_dict.population[:num_select]
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
            min(num_select, len(func))
        ):  # Ensure we don't select more than available
            chosen = np.random.choice(func, p=p)
            selected.append(chosen)

        return selected

    def _generate_single_operator_solution(
        self, prompt_content: List[dict], operator_type: str, sampler_id: int
    ) -> tuple[Solution, dict]:
        """Generate a single solution for an operator"""
        try:
            response, usage = self.config.running_llm.get_response(prompt_content)
            new_sol = self.config.interface.parse_response(response)
            self.verbose_info(
                f"Sampler {sampler_id}: Generated {operator_type} solution"
            )
            return new_sol, usage
        except Exception as e:
            self.verbose_info(
                f"Sampler {sampler_id}: Failed to generate {operator_type} solution - {str(e)}"
            )
            return Solution(""), {}

    def _get_run_state_class(self) -> Type[BaseRunStateDict]:
        return EoHRunStateDict
