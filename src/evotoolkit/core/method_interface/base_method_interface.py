# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import abc
from abc import abstractmethod

from ..base_task import BaseTask
from ..solution import Solution


class BaseMethodInterface(abc.ABC):
    """Base Adapter"""

    def __init__(self, task: BaseTask):
        self.task = task

    @abstractmethod
    def parse_response(self, response_str: str) -> Solution:
        raise NotImplementedError()
