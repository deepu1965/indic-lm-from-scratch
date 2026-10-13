"""
Phase 3.1 – Logical Generator package.

Public API
----------
from phase3.logical_generator import LogicalGenerator, GeneratorConfig, LogicalRecord
"""

from .config import GeneratorConfig
from .structures import LogicalRecord, Fact, Question, GroundTruth, EntityRef
from .generator import LogicalGenerator
from .splits import SplitManager
from .validator import LogicalValidator, ValidationResult

__all__ = [
    "GeneratorConfig",
    "LogicalRecord",
    "Fact",
    "Question",
    "GroundTruth",
    "EntityRef",
    "LogicalGenerator",
    "SplitManager",
    "LogicalValidator",
    "ValidationResult",
]
