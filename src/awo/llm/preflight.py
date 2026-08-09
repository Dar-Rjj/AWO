"""Minimal paid connectivity check for the configured model."""

from __future__ import annotations

from typing import Any

from awo.llm.openrouter import OpenRouterClient

PREFLIGHT_MARKER = "AWO_PREFLIGHT_OK"


def run_preflight(client: OpenRouterClient) -> dict[str, Any]:
    """Send one short deterministic request and return a secret-free summary."""

    result = client.chat(
        [
            {"role": "system", "content": "You are a connectivity checker."},
            {"role": "user", "content": f"Reply with exactly {PREFLIGHT_MARKER}"},
        ],
        temperature=0.0,
        max_tokens=32,
        metadata={"role": "preflight"},
    )
    return {
        "mode": "live",
        "ok": result.content.strip() == PREFLIGHT_MARKER,
        "expected": PREFLIGHT_MARKER,
        "received": result.content.strip(),
        "request_id": result.request_id,
        "response_id": result.response_id,
        "requested_model": result.requested_model,
        "actual_model": result.actual_model,
        "provider": result.provider,
        "attempts": result.attempts,
        "latency_seconds": result.latency_seconds,
        "usage": result.usage.model_dump(mode="json"),
        "request_sha256": result.request_sha256,
    }
