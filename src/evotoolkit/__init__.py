"""Composable program search over file artifacts."""

from importlib.metadata import PackageNotFoundError, version

from .core import EvaluationResult, EventLog, Instruction, Proposal, ProposalContext, Runner, Solution, TaskSpec

__author__ = "Ping Guo"
try:
    __version__ = version("evotoolkit")
except PackageNotFoundError:
    __version__ = "2.0.0.dev0"

__all__ = ["Runner", "TaskSpec", "EvaluationResult", "Solution", "Proposal", "ProposalContext", "Instruction", "EventLog", "__version__"]
