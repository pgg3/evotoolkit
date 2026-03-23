# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import concurrent.futures

from evotoolkit.core import IterativeMethod, Solution

from .programs_database import ProgramsDatabase
from .state import FunSearchState


class FunSearch(IterativeMethod):
    algorithm_name = "funsearch"
    history_layout = "batch"
    startup_title = "FUNSEARCH ALGORITHM STARTED"
    state_cls = FunSearchState

    def __init__(
        self,
        interface,
        *,
        running_llm,
        output_path: str = "./results",
        verbose: bool = True,
        max_sample_nums: int = 45,
        num_islands: int = 5,
        max_population_size: int = 1000,
        num_samplers: int = 5,
        num_evaluators: int = 5,
        programs_per_prompt: int = 2,
    ):
        self.max_sample_nums = max_sample_nums
        self.num_islands = num_islands
        self.max_population_size = max_population_size
        self.num_samplers = num_samplers
        self.num_evaluators = num_evaluators
        self.programs_per_prompt = programs_per_prompt

        super().__init__(
            interface=interface,
            output_path=output_path,
            running_llm=running_llm,
            verbose=verbose,
        )

    def _create_state(self) -> FunSearchState:
        return FunSearchState(
            task_spec=self.task.spec.copy(),
            batch_size=max(self.num_samplers, 1),
        )

    def prepare_initialization(self) -> None:
        if self.state.programs_database is None:
            self.state.programs_database = ProgramsDatabase(
                num_islands=self.num_islands,
                solutions_per_prompt=self.programs_per_prompt,
                reset_period=4 * 60 * 60,
            )
            existing_valid_solutions = [
                solution for solution in self.state.sol_history if solution.evaluation_res is not None and solution.evaluation_res.valid
            ]
            for solution in existing_valid_solutions:
                self.state.programs_database.register_solution(solution)
            if existing_valid_solutions:
                self.verbose_info(f"Rebuilt programs database from {len(existing_valid_solutions)} saved solutions")
            else:
                self.verbose_info("Initialized new programs database")

    def initialize_iteration(self) -> None:
        if self.state.sol_history or self.state.sample_count > 0:
            self.verbose_info(f"Continuing from sample {self.state.sample_count} with {len(self.state.sol_history)} solutions in history")

    def step_iteration(self) -> None:
        start_sample = self.state.sample_count + 1
        end_sample = self.state.sample_count + self.num_samplers
        self.verbose_info(f"Samples {start_sample} - {end_sample} / {self.max_sample_nums or 'unlimited'}")

        prompt_solutions, island_id = self.state.programs_database.get_prompt_solutions()
        if prompt_solutions:
            self.verbose_info(f"Selected {len(prompt_solutions)} solutions from island {island_id}")
        else:
            self.verbose_info(f"No prior programs available on island {island_id}; bootstrapping from the task prompt")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_samplers + self.num_evaluators) as executor:
            generate_futures = []
            eval_futures = []
            database_bootstrapped = self.state.programs_database.get_best_solution() is not None

            for sampler_id in range(self.num_samplers):
                generate_futures.append(executor.submit(self._generate_single_program, prompt_solutions, sampler_id))

            for future in concurrent.futures.as_completed(generate_futures):
                try:
                    new_program, usage = future.result()
                    self.state.usage_history["sample"].append(usage)
                    self.state.current_batch_usage.append(usage)
                    eval_futures.append((executor.submit(self.task.evaluate, new_program), new_program))
                except Exception as exc:
                    self.verbose_info(f"Program generation failed: {exc}")

            for eval_future, program in eval_futures:
                try:
                    program.evaluation_res = eval_future.result()
                    score_str = "None" if program.evaluation_res.score is None else f"{program.evaluation_res.score}"
                    self.verbose_info(f"Program evaluated - Score: {score_str}")
                except Exception as exc:
                    self.verbose_info(f"Program evaluation failed: {exc}")

                self.state.sol_history.append(program)
                self.state.current_batch_solutions.append(program)
                self.state.sample_count += 1

                if program.evaluation_res and program.evaluation_res.valid:
                    if not database_bootstrapped:
                        self.state.programs_database.register_solution(program)
                        database_bootstrapped = True
                        self.verbose_info("Bootstrapped all islands with the first valid program")
                    else:
                        self.state.programs_database.register_solution(program, island_id)
                    score_str = f"{program.evaluation_res.score:.6f}" if program.evaluation_res.score is not None else "None"
                    self.verbose_info(f"Registered valid program to island {island_id} (score: {score_str})")
                else:
                    self.verbose_info(f"Added invalid program to history (sample {self.state.sample_count})")

        best_solution = self.state.programs_database.get_best_solution()
        if best_solution and best_solution.evaluation_res:
            best_score_str = f"{best_solution.evaluation_res.score:.6f}" if best_solution.evaluation_res.score is not None else "None"
            self.verbose_info(f"Current best score: {best_score_str}")

        if self.state.sample_count % 50 == 0:
            stats = self.state.programs_database.get_statistics()
            self.verbose_info(
                f"Database stats: {stats['total_programs']} total programs, {stats['num_islands']} islands, best score: {stats['global_best_score']:.6f}"
            )

    def should_stop_iteration(self) -> bool:
        return self.state.status == "failed" or self.state.sample_count >= self.max_sample_nums

    def _select_best_solution(self) -> Solution | None:
        if self.state.programs_database is not None:
            return self.state.programs_database.get_best_solution()
        return self._get_best_sol(self.state.sol_history)

    def _save_artifacts(self) -> None:
        should_flush = len(self.state.current_batch_solutions) >= self.state.batch_size or self.state.status in {"failed", "completed"}
        if not self.state.current_batch_solutions or not should_flush:
            return

        batch_id = self.state.current_batch_id
        sample_range = (self.state.current_batch_start, self.state.sample_count)
        metadata = {
            "valid_count": sum(1 for s in self.state.current_batch_solutions if s.evaluation_res and s.evaluation_res.valid),
        }
        self.store.save_batch_history(
            batch_id=batch_id,
            sample_range=sample_range,
            solutions=self.state.current_batch_solutions,
            usage=self.state.current_batch_usage,
            metadata=metadata,
        )
        self.store.save_usage_history(self.state.usage_history)

        best_solution = self._select_best_solution()
        if best_solution and best_solution.evaluation_res:
            self.state.best_per_batch.append(
                {
                    "batch_id": batch_id,
                    "score": best_solution.evaluation_res.score,
                    "sol_string": best_solution.sol_string,
                }
            )
            self.store.save_summary("best_per_batch.json", self.state.best_per_batch)

        self.state.current_batch_start = self.state.sample_count
        self.state.current_batch_solutions = []
        self.state.current_batch_usage = []

    def _generate_single_program(self, prompt_solutions: list[Solution], sampler_id: int) -> tuple[Solution, dict]:
        try:
            prompt_content = self.interface.get_prompt(prompt_solutions)
            response, usage = self.running_llm.get_response(prompt_content)
            new_sol = self.interface.parse_response(response)
            self.verbose_info(f"Sampler {sampler_id}: Generated a program variant.")
            return new_sol, usage
        except Exception as exc:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate program - {exc}")
            return Solution(""), {}
