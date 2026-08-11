#!/usr/bin/env python3
"""Make one paid optimizer call and validate the returned workflow DSL."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from awo.aflow.dsl import OPERATOR_SETS, initial_candidate
from awo.aflow.optimizer import OpenRouterCandidateGenerator
from awo.aflow.search import RoundRecord, SearchContext
from awo.config import load_config
from awo.llm import JsonlRequestRecorder, client_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=sorted(OPERATOR_SETS))
    parser.add_argument("--record-file", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/openrouter_deepseek_chat.yaml"),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    parent = initial_candidate(args.dataset)
    parent_record = RoundRecord(
        round=1,
        parent_round=None,
        candidate_sha256=parent.sha256,
        modification=parent.modification,
        scores=(0.0,) * 5,
        mean_score=0.0,
        std_score=0.0,
        total_cost=0.0,
        total_tokens=0,
        generation_attempts=0,
        improved=True,
    )
    context = SearchContext(
        dataset=args.dataset,
        generated_index=1,
        round=2,
        parent=parent,
        parent_record=parent_record,
        parent_pool=(parent_record,),
        selection_probabilities=(1.0,),
        experience="No child workflow has been tried for this parent.",
        allowed_operators=tuple(sorted(OPERATOR_SETS[args.dataset])),
    )
    recorder = JsonlRequestRecorder(args.record_file) if args.record_file else None
    client = client_from_config(load_config(args.config), recorder=recorder)
    candidate = await OpenRouterCandidateGenerator(client).generate(context)
    print(json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
