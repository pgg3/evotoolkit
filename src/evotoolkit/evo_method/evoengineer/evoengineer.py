# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import concurrent.futures
from typing import List

from evotoolkit.core import PopulationMethod, Solution

from .state import EvoEngineerState


class EvoEngineer(PopulationMethod):
    algorithm_name = "evoengineer"
    startup_title = "EVOENGINEER ALGORITHM STARTED"
    state_cls = EvoEngineerState

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
        num_samplers: int = 4,
        num_evaluators: int = 4,
    ):
        self.max_generations = max_generations
        self.max_sample_nums = max_sample_nums
        self.pop_size = pop_size
        self.num_samplers = num_samplers
        self.num_evaluators = num_evaluators
        self.init_operators = interface.get_init_operators()
        self.offspring_operators = interface.get_offspring_operators()

        if not self.init_operators:
            raise ValueError("Adapter must provide at least one init operator")
        if not self.offspring_operators:
            raise ValueError("Adapter must provide at least one offspring operator")
        for op in self.init_operators:
            if op.selection_size != 0:
                raise ValueError(f"Init operator '{op.name}' must have selection_size=0, got {op.selection_size}")

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
        if len(valid_population) < self.interface.valid_require:
            self.verbose_info(
                f"The search is terminated since EvoEngineer unable to obtain {self.interface.valid_require} feasible algorithms during initialization."
            )
            self.state.status = "failed"
            return

        self.state.status = "running"

    def step_iteration(self) -> None:
        self.verbose_info(
            f"Generation {self.state.generation} - Sample {self.state.sample_count + 1} - "
            f"{self.state.sample_count + self.num_samplers} / {self.max_sample_nums or 'unlimited'}"
        )
        self._apply_operators_parallel(self.offspring_operators, f"Gen {self.state.generation}")
        self._trim_population(self.pop_size)
        self.state.generation += 1

    def should_stop_iteration(self) -> bool:
        return self.state.status == "failed" or self.state.generation >= self.max_generations or self.state.sample_count >= self.max_sample_nums

    def _initialize_population(self) -> None:
        self.verbose_info("Initializing population...")

        while self.state.sample_count < self.max_sample_nums:
            prev_sample_count = self.state.sample_count
            prev_valid_count = len(self._get_valid_population())

            self._apply_operators_parallel(self.init_operators, "Init")
            valid_count = len(self._get_valid_population())
            self.verbose_info(f"Valid solutions: {valid_count}/{self.pop_size}")

            self._persist_runtime()

            if valid_count >= self.interface.valid_require:
                break

            if self.state.sample_count == prev_sample_count and valid_count == prev_valid_count:
                self.verbose_info("Warning: Initialization made no progress; stopping early to avoid an infinite loop")
                break

        valid_population = self._get_valid_population()
        if len(valid_population) >= self.interface.valid_require:
            self.state.generation = 1
            self.verbose_info(f"Initialization completed with {len(valid_population)} valid solutions")
        else:
            self.verbose_info(f"Warning: Only {len(valid_population)} valid solutions obtained, need at least {self.interface.valid_require}")

    def _apply_operators_parallel(self, operators: List, generation_label: str = "") -> None:
        if not operators:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_samplers + self.num_evaluators) as executor:
            generate_futures = []
            eval_futures = []

            num_operators = len(operators)
            max_multiplier = self.num_samplers // num_operators
            target_samples = max_multiplier * num_operators
            samples_per_operator = target_samples // num_operators if num_operators else 0

            sample_id = 0
            for operator in operators:
                for _ in range(samples_per_operator):
                    selected_individuals = self._select_ranked_individuals(operator.selection_size)
                    future = executor.submit(
                        self._generate_single_solution,
                        operator,
                        selected_individuals,
                        sample_id,
                    )
                    generate_futures.append((operator.name, future))
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
                        self._register_population_solution(solution)
                        self.verbose_info(f"{operator_name} {generation_label} - Score: None (Invalid)")
                except Exception as exc:
                    self.verbose_info(f"Error generating {operator_name}: {exc}")

            eval_future_to_info = {eval_future: (solution, operator_name) for eval_future, solution, operator_name in eval_futures}
            for eval_future in concurrent.futures.as_completed([ef for ef, _, _ in eval_futures]):
                solution, operator_name = eval_future_to_info[eval_future]
                try:
                    solution.evaluation_res = eval_future.result()
                    score_str = "None" if solution.evaluation_res.score is None else f"{solution.evaluation_res.score}"
                    valid_str = "Valid" if solution.evaluation_res.valid else "Invalid"
                    self.verbose_info(f"{operator_name} {generation_label} - Score: {score_str} ({valid_str})")
                except Exception as exc:
                    self.verbose_info(f"Error evaluating {operator_name}: {exc}")
                self._register_population_solution(solution)

    def _generate_single_solution(self, operator, selected_individuals: List[Solution], sampler_id: int) -> tuple[Solution, dict]:
        try:
            current_best_sol = self._get_best_sol(self.state.population)
            random_descriptions = self._sample_random_descriptions(3)
            prompt_content = self.interface.get_operator_prompt(operator.name, selected_individuals, current_best_sol, random_descriptions)
            response, usage = self.running_llm.get_response(prompt_content)
            new_sol = self.interface.parse_response(response)
            self.verbose_info(f"Sampler {sampler_id}: Generated {operator.name} solution")
            return new_sol, usage
        except Exception as exc:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate {operator.name} solution - {exc}")
            return Solution(""), {}

    def _sample_random_descriptions(self, n: int) -> List[str]:
        import random

        descriptions = []
        for sol in self.state.population:
            if sol.metadata.description:
                descriptions.append(sol.metadata.description)

        if len(descriptions) <= n:
            return descriptions
        return random.sample(descriptions, n)
