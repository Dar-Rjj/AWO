from types import SimpleNamespace

from awo.aflow.official import (
    NEED_LLM_MARKER,
    RESULT_MARKER,
    OfficialSandboxRuntime,
    build_optimize_prompt,
    format_optimizer_prompt,
    initial_humaneval_candidate,
    parse_optimizer_response,
)
from awo.sandbox import PASS_MARKER, SandboxConfig, SandboxResult


class FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, messages, **_kwargs):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(
            content="def add(a, b):\n    return a + b",
            usage=SimpleNamespace(cost=0.001, total_tokens=9),
        )


class RelaySandbox:
    config = SandboxConfig()

    def __init__(self) -> None:
        self.sources: list[str] = []

    def run(self, source: str) -> SandboxResult:
        self.sources.append(source)
        if len(self.sources) == 1:
            stdout = NEED_LLM_MARKER + '{"prompt":"solve","index":0}\n' + PASS_MARKER
        else:
            stdout = (
                RESULT_MARKER
                + '{"prediction":"def add(a, b):\\n    return a + b","reported_cost":0.001}'
                + "\n"
                + PASS_MARKER
            )
        return SandboxResult("passed", 0, stdout, "", 0.01, "test-sandbox")


def test_official_optimizer_uses_upstream_xml_contract() -> None:
    candidate = initial_humaneval_candidate()
    prompt = format_optimizer_prompt(
        build_optimize_prompt(
            experience="No experience data found for round 1.",
            score=0.5,
            graph=candidate.graph,
            prompt=candidate.prompt,
            log="",
        )
    )
    assert "Only one detail point can be modified at a time" in prompt
    assert prompt.endswith(
        "<modification>modification</modification>\n<graph>graph</graph>\n<prompt>prompt</prompt>"
    )
    assert parse_optimizer_response(
        "<modification>x</modification><graph>g</graph><prompt>p</prompt>"
    ) == {"modification": "x", "graph": "g", "prompt": "p"}


def test_official_workflow_llm_call_is_relayed_out_of_sandbox() -> None:
    client = FakeClient()
    sandbox = RelaySandbox()
    runtime = OfficialSandboxRuntime(client, sandbox, humaneval_public_tests={})
    result = runtime.execute(
        initial_humaneval_candidate(),
        round_number=1,
        problem="def add(a, b): pass",
        entry_point="add",
    )
    assert result.prediction.endswith("return a + b")
    assert result.calls == 1
    assert result.sandbox_runs == 2
    assert client.prompts == ["solve"]
    assert all("class Workflow" in source for source in sandbox.sources)
    assert "OPENROUTER_API_KEY" not in "".join(sandbox.sources)
