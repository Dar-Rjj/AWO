"""Audited runtime shared by generated AFlow workflows and operators."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from awo.llm import ChatResult, OpenRouterClient
from awo.sandbox import DockerSandbox


class AFlowRuntime:
    """Bridge synchronous audited LLM calls into generated async workflows."""

    def __init__(
        self,
        client: OpenRouterClient,
        *,
        sandbox: DockerSandbox | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.sandbox = sandbox
        self.run_metadata = dict(run_metadata or {})
        self.responses: list[ChatResult] = []

    async def chat(
        self,
        prompt: str,
        *,
        operator: str,
        system: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request_metadata = {
            "method": "aflow",
            "role": "workflow_operator",
            "operator": operator,
            **self.run_metadata,
            **dict(metadata or {}),
        }
        result = await asyncio.to_thread(self.client.chat, messages, metadata=request_metadata)
        self.responses.append(result)
        return result

    @property
    def total_tokens(self) -> int:
        return sum(response.usage.total_tokens for response in self.responses)

    @property
    def total_cost(self) -> float:
        return sum(response.usage.cost or 0.0 for response in self.responses)

    def usage_summary(self) -> dict[str, float | int]:
        return {
            "call_count": len(self.responses),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }
