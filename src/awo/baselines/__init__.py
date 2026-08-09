"""Controlled implementations of the AFlow Table 1 baselines."""

from awo.baselines.cot import CoTBaseline, build_cot_prompt
from awo.baselines.cot_sc import CoTSCBaseline, build_selector_prompt, parse_selector_letter
from awo.baselines.io import IOBaseline, build_io_prompt, parse_io_response
from awo.baselines.medprompt import (
    MedPromptBaseline,
    build_medprompt_candidate_prompt,
    build_medprompt_voter_prompt,
    choose_vote_winner,
)
from awo.baselines.models import BaselineResult
from awo.baselines.multi_persona import (
    MultiPersonaBaseline,
    build_persona_prompt,
    build_synthesis_prompt,
    roles_for_dataset,
)
from awo.baselines.runner import score_baseline_result

__all__ = [
    "BaselineResult",
    "CoTBaseline",
    "CoTSCBaseline",
    "IOBaseline",
    "MedPromptBaseline",
    "MultiPersonaBaseline",
    "build_cot_prompt",
    "build_selector_prompt",
    "build_io_prompt",
    "build_medprompt_candidate_prompt",
    "build_medprompt_voter_prompt",
    "build_persona_prompt",
    "build_synthesis_prompt",
    "parse_io_response",
    "parse_selector_letter",
    "roles_for_dataset",
    "score_baseline_result",
    "choose_vote_winner",
]
