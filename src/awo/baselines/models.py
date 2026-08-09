"""Shared immutable result schema for all baseline methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from awo.llm import ChatResult


@dataclass(frozen=True)
class BaselineResult:
    method: str
    dataset: str
    sample_id: str
    prediction: str
    prompt_sha256: str
    responses: tuple[ChatResult, ...]
    protocol: str = "paper-faithful"

    @property
    def call_count(self) -> int:
        return len(self.responses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "prediction": self.prediction,
            "prompt_sha256": self.prompt_sha256,
            "protocol": self.protocol,
            "call_count": self.call_count,
            "responses": [response.model_dump(mode="json") for response in self.responses],
        }
