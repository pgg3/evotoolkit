# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""AI CUDA Engineer - standalone implementation with no inheritance."""

import os
import time

import numpy as np
from concurrent import futures

from .prompt_maker import PromptMaker
from .response_parser import ResponseParser
from .run_state_dict import AiCudaEngineerRunStateDict
from .run_config import AiCudaEngineerConfig


class AiCudaEngineer:
    """AI CUDA Engineer (no inheritance, completely standalone)."""

    def __init__(self, config: AiCudaEngineerConfig):
        self.config = config
        self.run_state_dict = self._load_run_state_dict()
        # Initialize history manager
        self.run_state_dict.init_history_manager(self.config.output_path)
        self._save_run_state_dict()

    # ===== Verbose methods (direct implementation) =====

    def verbose_info(self, message: str):
        if self.config.verbose:
            print(message)

    def verbose_title(self, text: str, total_width: int = 60):
        """Display a centered title with equal signs above and below"""
        if self.config.verbose:
            print("=" * total_width)
            print(text.center(total_width))
            print("=" * total_width)

    def verbose_stage(self, text: str, total_width: int = 60):
        """Display a stage separator with dashes"""
        if self.config.verbose:
            print("-" * total_width)
            print(text.center(total_width))
            print("-" * total_width)

    def verbose_gen(self, text: str, total_width: int = 60):
        """Display text centered with dashes on both sides"""
        if self.config.verbose:
            padding = (total_width - len(text)) // 2
            left_dashes = "-" * padding
            right_dashes = "-" * (total_width - len(text) - padding)
            print(left_dashes + text + right_dashes)

    # ===== State management methods (direct implementation) =====

    def _save_run_state_dict(self):
        """Save run state to file and history"""
        # Save history first
        self.run_state_dict.save_current_history()
        # Then save current state
        self.run_state_dict.to_json_file(os.path.join(self.config.output_path, "run_state.json"))

    def _load_run_state_dict(self) -> AiCudaEngineerRunStateDict:
        """Load run state from file"""
        if os.path.exists(os.path.join(self.config.output_path, "run_state.json")):
            self.verbose_info(f"Loading run state from file {os.path.join(self.config.output_path, 'run_state.json')}")
            return AiCudaEngineerRunStateDict.from_json_file(os.path.join(self.config.output_path, "run_state.json"))
        else:
            run_state_dict = AiCudaEngineerRunStateDict(self.config.task_info)
            self.verbose_info(f"Initialized run state dict.")
            return run_state_dict

    # ===== Main run method =====


    def run(self, hist_best_kernel_list):
        self.verbose_title("AI CUDA ENGINEER STARTED")

        while not self.run_state_dict.is_done:
            if self.run_state_dict.run_stage == "0":
                self.conversion()
            elif self.run_state_dict.run_stage == "1":
                self.translation()
            elif self.run_state_dict.run_stage == "2":
                self.evolution()
            elif self.run_state_dict.run_stage == "3":
                self.rag_evolution(hist_best_kernel_list)
            else:
                return

    def conversion(self):
        assert isinstance(self.config, AiCudaEngineerConfig)

        self.verbose_stage("Stage 0: Converting the function into functional code")
        if self.run_state_dict.task_info["func_py_code"] is None:
            parsed_convert_response = None
            error_msg = None
            error_restart = 0
            convert_success = False
            self.run_state_dict.usage_history["0"] = []
            self.verbose_info("Converting the function into functional code...")
            while not convert_success:
                convert_prompt = PromptMaker.make_convert_prompt(self.run_state_dict.task_info["org_py_code"], parsed_convert_response, error_msg)
                convert_response, convert_usage = self.config.conversion_llm.get_response(convert_prompt)
                parsed_convert_response = ResponseParser.parse_convert_response(convert_response)
                self.run_state_dict.usage_history["0"].append(convert_usage)
                self.run_state_dict.current_stage_usage.append(convert_usage)
                convert_result_dict = self.config.evaluator.compare_py_code_sandbox(self.run_state_dict.task_info["org_py_code"], parsed_convert_response)
                convert_success, error_msg = convert_result_dict["correctness"], convert_result_dict["error_msg"]

                if not convert_success:
                    error_restart += 1
                    self.verbose_info("Conversion failed, retrying...")

                if error_restart % 3 == 0:
                    error_msg = None

                if error_restart > 10:
                    self.run_state_dict.run_stage = "4"
                    self.run_state_dict.is_done = True
                    self._save_run_state_dict()
                    return
            self.verbose_info("Conversion successful!")
            self.run_state_dict.task_info["func_py_code"] = parsed_convert_response
            self.run_state_dict.run_stage = "1"
            self._save_run_state_dict()
        else:
            self.verbose_info("Conversion successful! (func_file already converted)")
            self.run_state_dict.run_stage = "1"
            self._save_run_state_dict()
            return

    def translation(self):
        assert isinstance(self.config, AiCudaEngineerConfig)

        self.verbose_stage("Stage 1: Translating the functional code into cuda code")
        if self.run_state_dict.task_info["cuda_code"] is None:
            self.verbose_info("Translating the functional code...")
            translate_success = False
            parsed_translate_response = None
            error_msg = None
            error_summary = None
            error_restart = 0
            self.run_state_dict.usage_history["1"] = []
            while not translate_success:
                if error_msg is not None:
                    error_summary_prompt = PromptMaker.make_translate_error_summary_prompt(
                        self.run_state_dict.task_info["func_py_code"],
                        parsed_translate_response,
                        error_msg
                    )
                    error_summary, error_summary_usage = self.config.translation_llm.get_response(error_summary_prompt)
                    self.run_state_dict.usage_history["1"].append(error_summary_usage)
                    self.run_state_dict.current_stage_usage.append(error_summary_usage)
                cuda_code_prompt = PromptMaker.make_translate_prompt(
                    self.run_state_dict.task_info["func_py_code"],
                    parsed_translate_response,
                    error_msg,
                    error_summary)
                translate_response, translate_usage = self.config.translation_llm.get_response(cuda_code_prompt)
                self.run_state_dict.usage_history["1"].append(translate_usage)
                self.run_state_dict.current_stage_usage.append(translate_usage)
                parsed_translate_response = ResponseParser.parse_translate_response(translate_response)
                evaluate_cuda_dict = self.config.evaluator.compare_func_cuda_sandbox(
                    self.run_state_dict.task_info["func_py_code"], parsed_translate_response
                )
                translate_success, error_msg = evaluate_cuda_dict["correctness"], evaluate_cuda_dict["error_msg"]

                if not translate_success:
                    error_restart += 1
                    self.verbose_info("Translation failed, retrying...")

                if error_restart % 3 == 0:
                    error_msg = None

                if error_restart > 10:
                    self.run_state_dict.run_stage = "4"
                    self.run_state_dict.is_done = True
                    self._save_run_state_dict()
                    return
            self.verbose_info("Translation successful!")
            self.run_state_dict.task_info["cuda_code"] = parsed_translate_response
            self.run_state_dict.run_stage = "2"
            self._save_run_state_dict()
        else:
            self.verbose_info("Translation successful! (cuda_file already converted)")
            self.run_state_dict.run_stage = "2"
            self._save_run_state_dict()
            return

    def evolution(self):
        assert isinstance(self.config, AiCudaEngineerConfig)

        self.verbose_stage("Stage 2: Evolving the cuda code")
        # make necessary task info, assuming that func and cuda code is right
        self.run_state_dict.task_info["func_runtime"] = self.run_state_dict.task_info.get(
            "func_runtime", self.config.evaluator.get_py_runtime_sandbox(self.run_state_dict.task_info["func_py_code"])["runtime"]
        )

        self._save_run_state_dict()

        self.verbose_info(
            f"Torch runtime: {self.run_state_dict.task_info['func_runtime']:.4f} ms, "
            f"CUDA runtime: {self.run_state_dict.task_info['cuda_info']['runtime']:.4f} ms"
        )

        if len(self.run_state_dict.optimization_history) == 0:
            initial_kernel = self.run_state_dict.task_info["cuda_info"]
            self.run_state_dict.add_optimization_result(initial_kernel)
            self._save_run_state_dict()

        for gen_i in range(self.run_state_dict.evo_gen_i, 10):
            self.verbose_gen(f"Gen {gen_i + 1}")
            top_5_kernel = self._get_valid_top_5_from_slow_to_fast(self.run_state_dict.optimization_history)
            best_kernel = self.run_state_dict.get_best_kernel()

            # Use best kernel or baseline for optimization
            cuda_individual = best_kernel if best_kernel else self.run_state_dict.task_info["cuda_info"]

            # Parallel processing with evo_llm_list
            llm_list_len = len(self.config.evo_llm_list)
            
            with futures.ThreadPoolExecutor(max_workers=llm_list_len) as executor:
                # Submit tasks to each LLM - each thread handles both propose AND evaluate
                future_to_index = {}
                for i in range(llm_list_len):
                    future = executor.submit(self._process_llm_proposal_and_evaluate, i, top_5_kernel, cuda_individual)
                    future_to_index[future] = i
                
                # Collect completed proposal+evaluation results
                for future in futures.as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        new_entry, usage = future.result()
                        self.run_state_dict.add_usage_result("2", usage)
                        if new_entry is not None:
                            self.run_state_dict.add_optimization_result(new_entry)
                            runtime_str = f"{new_entry['runtime']:.4f} ms" if new_entry['runtime'] is not None else "Failed"
                            self.verbose_info(f"LLM {i}: {new_entry['name']}, runtime: {runtime_str}, temp_str: {new_entry['temp_str']}")
                        else:
                            self.verbose_info(f"LLM {i}: Failed to generate valid proposal")

                    except Exception as e:
                        self.verbose_info(f"LLM {i}: Error processing proposal - {str(e)}")

            # Update generation info
            self.run_state_dict.evo_gen_i = gen_i + 1
            
            self.verbose_info(f"Generation {gen_i + 1} completed. Total entries: {len(self.run_state_dict.optimization_history)}")
            
            self._save_run_state_dict()

        self.run_state_dict.run_stage = "3"
        self._save_run_state_dict()

    def rag_evolution(self, hist_best_kernel_list):
        assert isinstance(self.config, AiCudaEngineerConfig)

        self.verbose_stage("Stage 3: RAG Evolving the cuda code")
        embedding_llm = self.config.embedding_llm
        if hist_best_kernel_list:
            embedding_database_list = [embedding_llm.get_embedding(term["task_info"]["func_py_code"]) for term in
                                       hist_best_kernel_list]
            current_embedding = embedding_llm.get_embedding(
                self.run_state_dict.task_info["func_py_code"]
            )

            embedding_database_list = np.array(embedding_database_list)  # N x 1536
            current_embedding = np.array(current_embedding)[np.newaxis, :]  # 1 x 1536

            # Calculate cosine similarity and get top-k similar codes
            similarities = [
                np.dot(current_embedding, db_emb.T) / (np.linalg.norm(current_embedding) * np.linalg.norm(db_emb))
                for db_emb in embedding_database_list]
            top_k_indices = np.argsort(similarities)[-5:][::-1].reshape(-1)  # Get top 5 similar codes
            similar_codes = [hist_best_kernel_list[i] for i in top_k_indices]
        else:
            similar_codes = []

        # Use similar pattern as evolution method but with RAG prompt
        best_kernel = self.run_state_dict.get_best_kernel()
        cuda_individual = best_kernel if best_kernel else self.run_state_dict.task_info["cuda_info"]

        RAG_TIMES = 5
        # Parallel processing with rag_llm for RAG evolution
        
        with futures.ThreadPoolExecutor(max_workers=RAG_TIMES) as executor:
            # Submit RAG_TIMES tasks - each thread handles both RAG propose AND evaluate
            future_to_index = {}
            for i in range(RAG_TIMES):
                future = executor.submit(self._process_rag_proposal_and_evaluate, i, similar_codes, cuda_individual)
                future_to_index[future] = i
            
            # Collect completed RAG proposal+evaluation results
            for future in futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    new_entry, usage = future.result()
                    self.run_state_dict.add_usage_result("3", usage)
                    if new_entry is not None:
                        self.run_state_dict.add_optimization_result(new_entry)
                        runtime_str = f"{new_entry['runtime']:.4f} ms" if new_entry['runtime'] is not None else "Failed"
                        self.verbose_info(f"RAG {i}: {new_entry['name']}, runtime: {runtime_str}, temp_str: {new_entry['temp_str']}")
                    else:
                        self.verbose_info(f"RAG {i}: Failed to generate valid proposal")
                except Exception as e:
                    self.verbose_info(f"RAG {i}: Error processing proposal - {str(e)}")

        self.run_state_dict.run_stage = "4"
        self.run_state_dict.is_done = True
        self._save_run_state_dict()
    
    def _process_rag_proposal_and_evaluate(self, task_index, similar_codes, cuda_individual):
        """Process a single RAG LLM proposal AND evaluate it completely in one thread."""
        # Use RAG prompt with similar codes as optimization history
        prompt = PromptMaker.make_rag_prompt(
            gpu_type=self.run_state_dict.task_info['gpu_type'],
            cuda_version=self.run_state_dict.task_info['cuda_version'],
            optimization_history=similar_codes,
            func_runtime=self.run_state_dict.task_info["func_runtime"],
            cuda_indiv=cuda_individual
        )
        return self._process_proposal_and_evaluate_common(task_index, prompt, "RAG", use_rag_llm=True)
    
    def _process_proposal_and_evaluate_common(self, llm_index, prompt, prompt_type, use_rag_llm=False):
        assert isinstance(self.config, AiCudaEngineerConfig)

        """Common function for processing LLM proposals and evaluating them."""
        try:
            # Step 1: Get response from LLM
            if use_rag_llm:
                response, usage = self.config.rag_llm.get_response(prompt)
            else:
                response, usage = self.config.evo_llm_list[llm_index].get_response(prompt)
            
            # Step 2: Parse the response
            parsed_response = ResponseParser.parse_evo_response(response)
            
            # Step 3: Create new entry structure
            new_entry = {
                "name": parsed_response["name"],
                "thought": parsed_response["thought"],
                "code": parsed_response["code"],
                "temp_str": None,
                "runtime": None,
                "prof_string": None,
                "compilation_error": False,
                "comparison_error": False,
                "error_msg": None
            }
            
            # Step 4: Evaluate CUDA code correctness
            cuda_comparison_result = self.config.evaluator.compare_func_cuda_sandbox(
                self.run_state_dict.task_info["func_py_code"],
                parsed_response["code"]
            )
            new_entry["temp_str"] = cuda_comparison_result.get("temp_str")
            new_entry["error_msg"] = cuda_comparison_result.get("error_msg")

            # Step 5: If correct, measure runtime performance
            if cuda_comparison_result["correctness"]:
                cuda_runtime_result = self.config.evaluator.get_cuda_runtime_sandbox(
                    self.run_state_dict.task_info["func_py_code"],
                    parsed_response["code"],
                    cuda_comparison_result.get("temp_str")
                )
                new_entry["runtime"] = cuda_runtime_result["runtime"]
                new_entry["prof_string"] = cuda_runtime_result["prof_string"]
                new_entry["error_msg"] = cuda_runtime_result.get("error_msg")
            else:
                new_entry["comparison_error"] = True
                new_entry["compilation_error"] = cuda_comparison_result["compilation_error"]
            
            return new_entry, usage
            
        except Exception as e:
            self.verbose_info(f"{prompt_type} LLM {llm_index}: Exception in proposal generation and evaluation - {str(e)}")
            return {
                "name": "failed_proposal",
                "thought": f"Failed due to exception: {str(e)}",
                "code": "",
                "temp_str": None,
                "runtime": None,
                "prof_string": None,
                "compilation_error": True,
                "comparison_error": True,
                "error_msg": "Unknown"
            }, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _get_valid_top_5_from_slow_to_fast(self, optimization_history):
        valid_individuals = []
        for individual in optimization_history:
            if individual.get("runtime") is not None and individual["runtime"] != float('inf'):
                valid_individuals.append(individual)
        
        # Step 2: Sort from quickest to slowest and take top 5
        valid_individuals.sort(key=lambda x: x["runtime"])
        top_5_fastest = valid_individuals[:5]
        
        # Step 3: Reverse to get from slow to fast (slowest of the top 5 first)
        top_5_slow_to_fast = top_5_fastest[::-1]
        
        return top_5_slow_to_fast

    def _process_llm_proposal_and_evaluate(self, llm_index, top_5_kernel, cuda_individual):
        """Process a single LLM proposal AND evaluate it completely in one thread."""
        # Use evolution prompt with top 5 kernels as optimization history
        prompt = PromptMaker.make_evo_prompt(
            self.run_state_dict.task_info['gpu_type'],
            self.run_state_dict.task_info['cuda_version'],
            top_5_kernel,
            self.run_state_dict.task_info["func_runtime"],
            cuda_individual
        )
        return self._process_proposal_and_evaluate_common(llm_index, prompt, "EVO")