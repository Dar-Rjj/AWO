"""Controlled implementations of the AFlow Table 1 baselines."""

from awo.baselines.cot import CoTBaseline, build_cot_prompt
from awo.baselines.cot_sc import CoTSCBaseline, build_selector_prompt, parse_selector_letter
from awo.baselines.io import IOBaseline, build_io_prompt, parse_io_response
from awo.baselines.models import BaselineResult
from awo.baselines.runner import score_baseline_result

__all__ = [
    "BaselineResult",
    "CoTBaseline",
    "CoTSCBaseline",
    "IOBaseline",
    "build_cot_prompt",
    "build_selector_prompt",
    "build_io_prompt",
    "parse_io_response",
    "parse_selector_letter",
    "score_baseline_result",
]
