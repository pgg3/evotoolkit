# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import concurrent.futures
from typing import List

from evotoolkit.core import PopulationMethod, Solution
from evotoolkit.registry import register_algorithm

from .state import EoHState


@register_algorithm("eoh")
class EoH(PopulationMethod):
    algorithm_name = "eoh"
    startup_title = "EOH ALGORITHM STARTED"
    state_cls = EoHState

    def __init__(
        self,
        interface,
        *,
        running_llm,
        output_path: str = "./results",
        verbose: bool = True,
        max_generations: int = 10,
        max_sample_nums: int = 45,
        pop_size: int = 5,
        selection_num: int = 2,
        use_e2_operator: bool = True,
        use_m1_operator: bool = True,
        use_m2_operator: bool = True,
        num_samplers: int = 5,
        num_evaluators: int = 5,
    ):
        self.max_generations = max_generations
        self.max_sample_nums = max_sample_nums
        self.pop_size = pop_size
        self.selection_num = selection_num
        self.use_e2_operator = use_e2_operator
        self.use_m1_operator = use_m1_operator
        self.use_m2_operator = use_m2_operator
        self.num_samplers = num_samplers
        self.num_evaluators = num_evaluators

        super().__init__(
            interface=interface,
            output_path=output_path,
            running_llm=running_llm,
            verbose=verbose,
        )

    def initialize_iteration(self) -> None:
        if self.state.generation == 0:
            self._initialize_population()

        valid_population = self._get_valid_population()
        if len(valid_population) < self.selection_num:
            self.verbose_info(f"The search is terminated since EoH unable to obtain {self.selection_num} feasible algorithms during initialization.")
            self.state.status = "failed"
            return

        self.state.status = "running"

    def step_iteration(self) -> None:
        self.verbose_info(
            f"Generation {self.state.generation} - Sample {self.state.sample_count + 1} - "
            f"{self.state.sample_count + self.num_samplers} / {self.max_sample_nums or 'unlimited'}"
        )

        new_solutions = self._apply_operators_parallel()
        for sol in new_solutions:
            self._register_population_solution(sol)

        self._trim_population(self.pop_size)
        self.state.generation += 1

    def should_stop_iteration(self) -> bool:
        return self.state.status == "failed" or self.state.generation >= self.max_generations or self.state.sample_count >= self.max_sample_nums

    def _initialize_population(self) -> None:
        self.verbose_info("Initializing population...")

        while len(self._get_valid_population()) < self.pop_size and self.state.sample_count < self.max_sample_nums:
            evaluated_solutions = self._generate_and_evaluate_initial_solutions()
            for sol in evaluated_solutions:
                self._register_population_solution(sol)

                score_str = "None" if not sol.evaluation_res or sol.evaluation_res.score is None else f"{sol.evaluation_res.score}"
                valid_str = "Valid" if sol.evaluation_res and sol.evaluation_res.valid else "Invalid"
                self.verbose_info(f"Initial sample {self.state.sample_count} - Score: {score_str} ({valid_str})")

            valid_count = len(self._get_valid_population())
            self.verbose_info(f"Valid solutions: {valid_count}/{self.pop_size}")

            self._persist_runtime()

        valid_population = self._get_valid_population()
        if len(valid_population) >= self.selection_num:
            self.state.generation = 1
            self.verbose_info(f"Initialization completed with {len(valid_population)} valid solutions")
        else:
            self.verbose_info(f"Warning: Only {len(valid_population)} valid solutions obtained, need at least {self.selection_num}")

    def _generate_and_evaluate_initial_solutions(self) -> List[Solution]:
        evaluated_solutions = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_samplers + self.num_evaluators) as executor:
            generate_futures = []
            eval_futures = []

            for sampler_id in range(self.num_samplers):
                generate_futures.append(executor.submit(self._generate_single_initial_solution, sampler_id))

            for future in concurrent.futures.as_completed(generate_futures):
                try:
                    new_sol, usage = future.result()
                    self._record_generation_usage(usage)

                    if new_sol.sol_string.strip():
                        eval_futures.append((executor.submit(self.task.evaluate, new_sol), new_sol))
                    else:
                        evaluated_solutions.append(new_sol)
                except Exception as exc:
                    self.verbose_info(f"Initial solution generation failed: {exc}")

            for eval_future, solution in eval_futures:
                try:
                    solution.evaluation_res = eval_future.result()
                except Exception as exc:
                    self.verbose_info(f"Evaluation failed: {exc}")
                evaluated_solutions.append(solution)

        return evaluated_solutions

    def _generate_single_initial_solution(self, sampler_id: int) -> tuple[Solution, dict]:
        try:
            prompt_content = self.interface.get_prompt_i1()
            response, usage = self.running_llm.get_response(prompt_content)
            new_sol = self.interface.parse_response(response)
            self.verbose_info(f"Sampler {sampler_id}: Generated initial solution")
            return new_sol, usage
        except Exception as exc:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate initial solution - {exc}")
            return Solution(""), {}

    def _apply_operators_parallel(self) -> List[Solution]:
        new_solutions = []
        operator_tasks = []

        selected_individuals = self._select_ranked_individuals(self.selection_num)
        if selected_individuals:
            operator_tasks.append(("E1", self.interface.get_prompt_e1(selected_individuals)))

        if self.use_e2_operator:
            selected_individuals = self._select_ranked_individuals(self.selection_num)
            if selected_individuals:
                operator_tasks.append(("E2", self.interface.get_prompt_e2(selected_individuals)))

        if self.use_m1_operator:
            selected_individuals = self._select_ranked_individuals(1)
            if selected_individuals:
                operator_tasks.append(("M1", self.interface.get_prompt_m1(selected_individuals[0])))

        if self.use_m2_operator:
            selected_individuals = self._select_ranked_individuals(1)
            if selected_individuals:
                operator_tasks.append(("M2", self.interface.get_prompt_m2(selected_individuals[0])))

        if not operator_tasks:
            return new_solutions

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_samplers + self.num_evaluators) as executor:
            generate_futures = []
            eval_futures = []

            num_operators = len(operator_tasks)
            max_multiplier = self.num_samplers // num_operators
            target_samples = max_multiplier * num_operators
            samples_per_operator = target_samples // num_operators if num_operators else 0

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

            future_to_operator = {future: operator_name for operator_name, future in generate_futures}
            for future in concurrent.futures.as_completed([f for _, f in generate_futures]):
                operator_name = future_to_operator[future]
                try:
                    solution, usage = future.result()
                    self._record_generation_usage(usage)

                    if solution.sol_string.strip():
                        eval_futures.append((executor.submit(self.task.evaluate, solution), solution, operator_name))
                    else:
                        new_solutions.append(solution)
                        self.verbose_info(f"{operator_name} Gen {self.state.generation} - Score: None (Invalid)")
                except Exception as exc:
                    self.verbose_info(f"Error generating {operator_name}: {exc}")

            for eval_future, solution, operator_name in eval_futures:
                try:
                    solution.evaluation_res = eval_future.result()
                    score_str = "None" if solution.evaluation_res.score is None else f"{solution.evaluation_res.score}"
                    valid_str = "Valid" if solution.evaluation_res.valid else "Invalid"
                    self.verbose_info(f"{operator_name} Gen {self.state.generation} - Score: {score_str} ({valid_str})")
                except Exception as exc:
                    self.verbose_info(f"Error evaluating {operator_name}: {exc}")
                new_solutions.append(solution)

        return new_solutions

    def _generate_single_operator_solution(self, prompt_content: List[dict], operator_type: str, sampler_id: int) -> tuple[Solution, dict]:
        try:
            response, usage = self.running_llm.get_response(prompt_content)
            new_sol = self.interface.parse_response(response)
            self.verbose_info(f"Sampler {sampler_id}: Generated {operator_type} solution")
            return new_sol, usage
        except Exception as exc:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate {operator_type} solution - {exc}")
            return Solution(""), {}
