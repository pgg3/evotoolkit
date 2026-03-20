# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import concurrent.futures
from typing import List

from evotoolkit.core import Method, Solution
from evotoolkit.registry import register_algorithm

from .state import EoHState


@register_algorithm("eoh")
class EoH(Method):
    algorithm_name = "eoh"
    history_layout = "generation"

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

    def _create_state(self) -> EoHState:
        return EoHState(task_info=dict(self.task.task_info))

    def _bootstrap(self) -> None:
        self.verbose_title("EOH ALGORITHM STARTED")

        if not self.state.sol_history:
            init_sol = self._create_seed_solution()
            if init_sol is None:
                self.state.status = "failed"
                return
            self.state.sol_history.append(init_sol)
            self.state.population.append(init_sol)
            score = init_sol.evaluation_res.score if init_sol.evaluation_res else "None"
            self.verbose_info(f"Initialized with baseline solution (score: {score})")

        if self.state.generation == 0:
            self._initialize_population()

        valid_population = self._get_valid_population(self.state.population)
        if len(valid_population) < self.selection_num:
            self.verbose_info(
                f"The search is terminated since EoH unable to obtain {self.selection_num} feasible algorithms during initialization."
            )
            self.state.status = "failed"
            return

        self.state.status = "running"

    def _step(self) -> None:
        self.verbose_info(
            f"Generation {self.state.generation} - Sample {self.state.tot_sample_nums + 1} - "
            f"{self.state.tot_sample_nums + self.num_samplers} / {self.max_sample_nums or 'unlimited'}"
        )

        new_solutions = self._apply_operators_parallel()
        for sol in new_solutions:
            self.state.sol_history.append(sol)
            self.state.population.append(sol)
            self.state.current_generation_solutions.append(sol)
            self.state.tot_sample_nums += 1

        self._manage_population_size()
        self.state.generation += 1

    def _should_stop(self) -> bool:
        return (
            self.state.status == "failed"
            or self.state.generation >= self.max_generations
            or self.state.tot_sample_nums >= self.max_sample_nums
        )

    def _select_best_solution(self) -> Solution | None:
        return self._get_best_sol(self.state.sol_history)

    def _save_artifacts(self) -> None:
        if not self.state.current_generation_solutions:
            return

        valid_sols = [s for s in self.state.current_generation_solutions if s.evaluation_res and s.evaluation_res.valid]
        statistics = {
            "total_solutions": len(self.state.current_generation_solutions),
            "valid_solutions": len(valid_sols),
            "valid_rate": len(valid_sols) / len(self.state.current_generation_solutions),
        }
        if valid_sols:
            scores = [s.evaluation_res.score for s in valid_sols]
            statistics["avg_score"] = sum(scores) / len(scores)
            statistics["best_score"] = max(scores)
            statistics["worst_score"] = min(scores)

        completed_generation = max(self.state.generation - 1, 0)
        self.store.save_generation_history(
            generation=completed_generation,
            solutions=self.state.current_generation_solutions,
            usage=self.state.current_generation_usage,
            statistics=statistics,
        )
        self.store.save_usage_history(self.state.usage_history)

        best_solution = self._select_best_solution()
        if best_solution and best_solution.evaluation_res:
            self.state.best_per_generation.append(
                {
                    "generation": completed_generation,
                    "score": best_solution.evaluation_res.score,
                    "sol_string": best_solution.sol_string,
                }
            )
            self.store.save_best_per_generation(self.state.best_per_generation)

        self.state.current_generation_solutions = []
        self.state.current_generation_usage = []

    def _initialize_population(self) -> None:
        self.verbose_info("Initializing population...")

        while (
            len(self._get_valid_population(self.state.population)) < self.pop_size
            and self.state.tot_sample_nums < self.max_sample_nums
        ):
            evaluated_solutions = self._generate_and_evaluate_initial_solutions()
            for sol in evaluated_solutions:
                self.state.sol_history.append(sol)
                self.state.population.append(sol)
                self.state.current_generation_solutions.append(sol)
                self.state.tot_sample_nums += 1

                score_str = "None" if not sol.evaluation_res or sol.evaluation_res.score is None else f"{sol.evaluation_res.score}"
                valid_str = "Valid" if sol.evaluation_res and sol.evaluation_res.valid else "Invalid"
                self.verbose_info(f"Initial sample {self.state.tot_sample_nums} - Score: {score_str} ({valid_str})")

            valid_count = len(self._get_valid_population(self.state.population))
            self.verbose_info(f"Valid solutions: {valid_count}/{self.pop_size}")

            self._persist_runtime()

        valid_population = self._get_valid_population(self.state.population)
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
                    self.state.usage_history["sample"].append(usage)
                    self.state.current_generation_usage.append(usage)

                    if new_sol.sol_string.strip():
                        eval_futures.append((executor.submit(self.task.evaluate_code, new_sol.sol_string), new_sol))
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

    def _get_valid_population(self, population: List[Solution]) -> List[Solution]:
        return [sol for sol in population if sol.evaluation_res and sol.evaluation_res.valid]

    def _apply_operators_parallel(self) -> List[Solution]:
        new_solutions = []
        operator_tasks = []

        selected_individuals = self._select_individuals(self.selection_num)
        if selected_individuals:
            operator_tasks.append(("E1", self.interface.get_prompt_e1(selected_individuals)))

        if self.use_e2_operator:
            selected_individuals = self._select_individuals(self.selection_num)
            if selected_individuals:
                operator_tasks.append(("E2", self.interface.get_prompt_e2(selected_individuals)))

        if self.use_m1_operator:
            selected_individuals = self._select_individuals(1)
            if selected_individuals:
                operator_tasks.append(("M1", self.interface.get_prompt_m1(selected_individuals[0])))

        if self.use_m2_operator:
            selected_individuals = self._select_individuals(1)
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
                    self.state.usage_history["sample"].append(usage)
                    self.state.current_generation_usage.append(usage)

                    if solution.sol_string.strip():
                        eval_futures.append((executor.submit(self.task.evaluate_code, solution.sol_string), solution, operator_name))
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

    def _manage_population_size(self) -> None:
        if len(self.state.population) <= self.pop_size:
            return

        valid_solutions = self._get_valid_population(self.state.population)
        invalid_solutions = [sol for sol in self.state.population if sol not in valid_solutions]

        valid_solutions.sort(
            key=lambda x: x.evaluation_res.score if x.evaluation_res and x.evaluation_res.score is not None else float("-inf"),
            reverse=True,
        )

        new_population = valid_solutions[: min(len(valid_solutions), self.pop_size)]

        remaining_slots = self.pop_size - len(new_population)
        if remaining_slots > 0 and invalid_solutions:
            new_population.extend(invalid_solutions[-remaining_slots:])

        self.state.population = new_population
        valid_count = len(self._get_valid_population(new_population))
        self.verbose_info(f"Population managed: {len(new_population)} total ({valid_count} valid, {len(new_population) - valid_count} invalid)")

    def _select_individuals(self, num_select: int) -> List[Solution]:
        import math

        import numpy as np

        if num_select <= 0:
            return []

        funcs = [
            sol
            for sol in self.state.population
            if sol.evaluation_res
            and sol.evaluation_res.valid
            and sol.evaluation_res.score is not None
            and not math.isinf(sol.evaluation_res.score)
            and not math.isnan(sol.evaluation_res.score)
        ]

        if not funcs:
            return self.state.population[:num_select] if self.state.population else []

        ranked = sorted(funcs, key=lambda f: f.evaluation_res.score, reverse=True)
        probabilities = np.array([1 / (r + len(ranked)) for r in range(len(ranked))])
        probabilities = probabilities / np.sum(probabilities)

        selected = []
        for _ in range(min(num_select, len(ranked))):
            selected.append(np.random.choice(ranked, p=probabilities))
        return selected

    def _generate_single_operator_solution(self, prompt_content: List[dict], operator_type: str, sampler_id: int) -> tuple[Solution, dict]:
        try:
            response, usage = self.running_llm.get_response(prompt_content)
            new_sol = self.interface.parse_response(response)
            self.verbose_info(f"Sampler {sampler_id}: Generated {operator_type} solution")
            return new_sol, usage
        except Exception as exc:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate {operator_type} solution - {exc}")
            return Solution(""), {}
