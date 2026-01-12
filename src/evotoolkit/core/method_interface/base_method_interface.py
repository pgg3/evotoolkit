# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import abc
from abc import abstractmethod

from ..base_task import BaseTask
from ..solution import Solution


class BaseMethodInterface(abc.ABC):
    """Base interface for method-specific prompt building and response parsing.

    Interface is responsible for:
    - Building prompts for the LLM based on solutions and task info
    - Parsing LLM responses into Solution objects

    Interface is NOT responsible for:
    - Creating initial solutions (handled by evolution methods or task)
    - Evaluating solutions (handled by task)
    """

    def __init__(self, task: BaseTask):
        self.task = task

    @abstractmethod
    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response string into a Solution object."""
        raise NotImplementedError()
