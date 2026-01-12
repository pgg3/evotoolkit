# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import concurrent.futures
from typing import Type

from evotoolkit.core import BaseRunStateDict, Method, Solution
from evotoolkit.registry import register_algorithm

from .programs_database import ProgramsDatabase
from .run_config import FunSearchConfig
from .run_state_dict import FunSearchRunStateDict


@register_algorithm("funsearch", config=FunSearchConfig)
class FunSearch(Method):
    def __init__(self, config: FunSearchConfig):
        super().__init__(config)
        self.config = config

    def run(self):
        """Main FunSearch algorithm execution"""
        self.verbose_title("FUNSEARCH ALGORITHM STARTED")

        # Initialize usage history
        if "sample" not in self.run_state_dict.usage_history:
            self.run_state_dict.usage_history["sample"] = []

        # Initialize or restore programs database
        if self.run_state_dict.has_database_state(self.config.output_path):
            # Restore from saved database file
            database_dict = self.run_state_dict.load_database_state(
                self.config.output_path
            )
            if database_dict:
                programs_db = ProgramsDatabase.from_dict(database_dict)
                self.verbose_info("Restored programs database from saved state")
            else:
                # Failed to load, create new database
                programs_db = ProgramsDatabase(
                    num_islands=self.config.num_islands,
                    solutions_per_prompt=self.config.programs_per_prompt,
                    reset_period=4 * 60 * 60,  # 4 hours
                )
                self.verbose_info("Failed to restore database, initialized new one")
        else:
            # Initialize new database
            programs_db = ProgramsDatabase(
                num_islands=self.config.num_islands,
                solutions_per_prompt=self.config.programs_per_prompt,
                reset_period=4 * 60 * 60,  # 4 hours
            )
            self.verbose_info("Initialized new programs database")

        # Initialize with seed program if sol_history is empty
        if len(self.run_state_dict.sol_history) == 0:
            self.verbose_info("Starting with empty solution history - will generate initial solutions in main loop")
        else:
            self.verbose_info(
                f"Continuing from sample {self.run_state_dict.tot_sample_nums} with {len(self.run_state_dict.sol_history)} solutions in history"
            )

            # Rebuild database from sol_history if needed
            if not self.run_state_dict.has_database_state(self.config.output_path):
                self.verbose_info("Rebuilding database from solution history...")
                for solution in self.run_state_dict.sol_history:
                    if solution.evaluation_res and solution.evaluation_res.valid:
                        programs_db.register_solution(solution)

        # Main sampling loop
        while self.run_state_dict.tot_sample_nums < self.config.max_sample_nums:
            try:
                start_sample = self.run_state_dict.tot_sample_nums + 1
                end_sample = (
                    self.run_state_dict.tot_sample_nums + self.config.num_samplers
                )
                self.verbose_info(
                    f"Samples {start_sample} - {end_sample} / {self.config.max_sample_nums or 'unlimited'}"
                )

                # Get prompt solutions from random island (may be empty for initial generation)
                prompt_solutions, island_id = programs_db.get_prompt_solutions()
                if not prompt_solutions:
                    self.verbose_info("No solutions in database - generating from scratch")
                    # Use empty list, interface should handle this case
                    prompt_solutions = []
                    island_id = 0
                else:
                    self.verbose_info(
                        f"Selected {len(prompt_solutions)} solutions from island {island_id}"
                    )

                # Async generate and evaluate programs - single executor for both
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.num_samplers + self.config.num_evaluators
                ) as executor:
                    # Submit all generate tasks
                    generate_futures = []
                    eval_futures = []

                    for sampler_id in range(self.config.num_samplers):
                        future = executor.submit(
                            self._generate_single_program, prompt_solutions, sampler_id
                        )
                        generate_futures.append(future)

                    # Process generated programs as they complete and immediately submit for evaluation
                    for future in concurrent.futures.as_completed(generate_futures):
                        try:
                            new_program, usage = future.result()
                            self.run_state_dict.usage_history["sample"].append(usage)
                            self.run_state_dict.current_batch_usage.append(
                                usage
                            )  # 添加到当前批次usage历史

                            # Immediately submit for evaluation without waiting
                            eval_future = executor.submit(
                                self.config.task.evaluate_code, new_program.sol_string
                            )
                            eval_futures.append((eval_future, new_program))
                        except Exception as e:
                            self.verbose_info(f"Program generation failed: {str(e)}")
                            continue

                    # Collect evaluation results as they complete
                    for eval_future, program in eval_futures:
                        try:
                            evaluation_res = eval_future.result()
                            program.evaluation_res = evaluation_res
                            score_str = (
                                "None"
                                if evaluation_res.score is None
                                else f"{evaluation_res.score}"
                            )
                            self.verbose_info(f"Program evaluated - Score: {score_str}")
                        except Exception as e:
                            self.verbose_info(f"Program evaluation failed: {str(e)}")
                            continue

                        # Process each evaluated program immediately
                        # Add ALL programs (valid/invalid) to sol_history
                        self.run_state_dict.sol_history.append(program)
                        self.run_state_dict.current_batch_solutions.append(
                            program
                        )  # 添加到当前批次历史
                        self.run_state_dict.tot_sample_nums += 1

                        # Only register valid programs to the database/island
                        if program.evaluation_res and program.evaluation_res.valid:
                            programs_db.register_solution(program, island_id)

                            score_str = (
                                f"{program.evaluation_res.score:.6f}"
                                if program.evaluation_res.score is not None
                                else "None"
                            )
                            self.verbose_info(
                                f"Registered valid program to island {island_id} (score: {score_str})"
                            )
                        else:
                            self.verbose_info(
                                f"Added invalid program to history (sample {self.run_state_dict.tot_sample_nums})"
                            )

                # Log current best
                best_solution = programs_db.get_best_solution()
                if best_solution and best_solution.evaluation_res:
                    best_score_str = (
                        f"{best_solution.evaluation_res.score:.6f}"
                        if best_solution.evaluation_res.score is not None
                        else "None"
                    )
                    self.verbose_info(f"Current best score: {best_score_str}")

                # Show database statistics periodically
                if self.run_state_dict.tot_sample_nums % 50 == 0:
                    stats = programs_db.get_statistics()
                    self.verbose_info(
                        f"Database stats: {stats['total_programs']} total programs, {stats['num_islands']} islands, best score: {stats['global_best_score']:.6f}"
                    )

                self._save_run_state_dict_with_database(programs_db)

            except KeyboardInterrupt:
                self.verbose_info("Interrupted by user")
                break
            except Exception as e:
                self.verbose_info(f"Sampling error: {str(e)}")
                continue

        # Mark as done and save final state with database
        self.run_state_dict.is_done = True
        self._save_run_state_dict_with_database(programs_db)

        # Log final statistics
        final_stats = programs_db.get_statistics()
        self.verbose_info(
            f"Final stats: {final_stats['total_programs']} programs, best score: {final_stats['global_best_score']:.6f}"
        )

    def _save_run_state_dict_with_database(self, programs_db):
        """Override base method to also save database state"""
        # Save database state first
        self.run_state_dict.save_database_state(
            programs_db.to_dict(), self.config.output_path
        )
        # Then save run state as usual
        super()._save_run_state_dict()

        # Show database file location
        self.verbose_info(
            f"Programs database saved to: {self.run_state_dict.database_file}"
        )

    def _generate_single_program(
        self, prompt_solutions: list[Solution], sampler_id: int
    ) -> tuple[Solution, dict]:
        """Generate single program variant using LLM based on prompt solutions"""
        try:
            # Get prompt from adapter based on selected solutions
            prompt_content = self.config.interface.get_prompt(prompt_solutions)

            response, usage = self.config.running_llm.get_response(prompt_content)

            new_sol = self.config.interface.parse_response(response)
            self.verbose_info(f"Sampler {sampler_id}: Generated a program variant.")
            return new_sol, usage
        except Exception as e:
            self.verbose_info(
                f"Sampler {sampler_id}: Failed to generate program - {str(e)}"
            )
            return Solution(""), {}  # Return usage even if failed

    def _get_run_state_class(self) -> Type[BaseRunStateDict]:
        return FunSearchRunStateDict
