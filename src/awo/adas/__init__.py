"""Controlled, protocol-compatible ADAS Meta Agent Search."""

from awo.adas.dsl import (
    ADASArchitecture,
    ADASValidationError,
    execute_architecture,
    initial_archive,
    validate_architecture,
)
from awo.adas.evaluation import ADASValidationEvaluator
from awo.adas.executor import build_adas_task, run_architecture
from awo.adas.meta import TASK_DESCRIPTIONS, ADASMetaGenerator
from awo.adas.search import (
    ADASConfig,
    ADASObservation,
    ADASRecord,
    ADASSearch,
    ADASSummary,
)

__all__ = [
    "ADASArchitecture",
    "ADASConfig",
    "ADASMetaGenerator",
    "ADASObservation",
    "ADASRecord",
    "ADASSearch",
    "ADASSummary",
    "ADASValidationError",
    "ADASValidationEvaluator",
    "TASK_DESCRIPTIONS",
    "build_adas_task",
    "execute_architecture",
    "initial_archive",
    "run_architecture",
    "validate_architecture",
]
