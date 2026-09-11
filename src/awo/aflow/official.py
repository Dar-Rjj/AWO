"""Paper-commit AFlow Python workflow adapter.

The optimizer prompt and generated graph are taken from the pinned upstream
snapshot. Untrusted graph execution happens in the configured sandbox. LLM
requests are relayed back to the host without exposing credentials to it.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
import textwrap
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from awo.llm import OpenRouterClient
from awo.sandbox import SandboxRunner

PAPER_COMMIT = "289d5544cfb5698799b207b931a392aadc9e9c1a"
SNAPSHOT_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "aflow_paper"
NEED_LLM_MARKER = "__AFLOW_NEED_LLM__"
RESULT_MARKER = "__AFLOW_RESULT__"
ERROR_MARKER = "__AFLOW_ERROR__"


def _source(relative: str) -> str:
    return (SNAPSHOT_ROOT / relative).read_text(encoding="utf-8")


def _literal_assignments(relative: str) -> dict[str, str]:
    tree = ast.parse(_source(relative))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                values[target.id] = value
    return values


def verify_snapshot() -> None:
    commit = (SNAPSHOT_ROOT / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()
    if commit != PAPER_COMMIT:
        raise RuntimeError(f"AFlow snapshot commit mismatch: {commit!r}")


@dataclass(frozen=True)
class OfficialPythonCandidate:
    """The three fields emitted by upstream GraphOptimize."""

    modification: str
    graph: str
    prompt: str
    parent_round: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OfficialPythonCandidate:
        return cls(
            modification=str(value["modification"]),
            graph=str(value["graph"]),
            prompt=str(value["prompt"]),
            parent_round=(int(value["parent_round"]) if value.get("parent_round") else None),
            metadata=dict(value.get("metadata", {})),
        )


def initial_humaneval_candidate() -> OfficialPythonCandidate:
    verify_snapshot()
    graph_file = _source("workspace/HumanEval/workflows/round_1/graph.py")
    matches = re.findall(r"class Workflow:.+", graph_file, re.DOTALL)
    if len(matches) != 1:
        raise RuntimeError("upstream initial HumanEval graph cannot be extracted")
    return OfficialPythonCandidate(
        modification="",
        graph=matches[0],
        prompt=_source("workspace/HumanEval/workflows/round_1/prompt.py"),
        parent_round=None,
        metadata={"upstream_commit": PAPER_COMMIT},
    )


def operator_description(names: list[str]) -> str:
    operators = json.loads(_source("workspace/HumanEval/workflows/template/operator.json"))
    lines = []
    for index, name in enumerate(names, start=1):
        item = operators[name]
        lines.append(
            f"{index}. {name}: {item['description']}, with interface {item['interface']})."
        )
    return "\n".join(lines) + "\n"


def build_optimize_prompt(
    *, experience: str, score: float, graph: str, prompt: str, log: str
) -> str:
    """Reproduce GraphUtils.create_graph_optimize_prompt from the pinned source."""

    values = _literal_assignments("scripts/prompts/optimize_prompt.py")
    graph_input = values["WORKFLOW_INPUT"].format(
        experience=experience,
        score=score,
        graph=graph,
        prompt=prompt,
        operator_description=operator_description(
            ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]
        ),
        type="code",
        log=log,
    )
    graph_system = values["WORKFLOW_OPTIMIZE_PROMPT"].format(type="code")
    return graph_input + values["WORKFLOW_CUSTOM_USE"] + graph_system


def format_optimizer_prompt(prompt: str) -> str:
    """Reproduce XmlFormatter.from_model(GraphOptimize).prepare_prompt."""

    return prompt + (
        "\n# Response format (must be strictly followed) (do not include any other "
        "formats except for the given XML format):\n"
        "<modification>modification</modification>\n"
        "<graph>graph</graph>\n"
        "<prompt>prompt</prompt>"
    )


def parse_optimizer_response(response: str) -> dict[str, str]:
    """Reproduce upstream XmlFormatter.validate_response field extraction."""

    matches = re.findall(r"<(\w+)>(.*?)</\1>", response, re.DOTALL)
    return {name: value.strip() for name, value in matches}


class OfficialCandidateGenerator:
    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    async def generate(
        self,
        *,
        parent: OfficialPythonCandidate,
        parent_round: int,
        score: float,
        experience: str,
        log: str,
        round_number: int,
    ) -> OfficialPythonCandidate:
        optimizer_prompt = format_optimizer_prompt(
            build_optimize_prompt(
                experience=experience,
                score=score,
                graph=parent.graph,
                prompt=parent.prompt,
                log=log,
            )
        )
        result = await asyncio.to_thread(
            self.client.chat,
            [{"role": "user", "content": optimizer_prompt}],
            metadata={
                "role": "aflow_optimizer",
                "dataset": "HumanEval",
                "round": round_number,
                "parent_round": parent_round,
                "upstream_commit": PAPER_COMMIT,
            },
        )
        fields = parse_optimizer_response(result.content)
        # Upstream directly indexes all three fields after its lenient formatter.
        return OfficialPythonCandidate(
            modification=fields["modification"],
            graph=fields["graph"],
            prompt=fields["prompt"],
            parent_round=parent_round,
            metadata={
                "request_sha256": result.request_sha256,
                "provider": result.provider,
                "cost": result.usage.cost,
                "tokens": result.usage.total_tokens,
                "upstream_commit": PAPER_COMMIT,
            },
        )


@dataclass(frozen=True)
class OfficialExecution:
    prediction: str
    reported_cost: float
    relay_cost: float
    tokens: int
    calls: int
    sandbox_runs: int


_WORKER = r'''
import ast
import asyncio
import json
import re
import sys
import traceback
import types

NEED = "__AFLOW_NEED_LLM__"
RESULT = "__AFLOW_RESULT__"
ERROR = "__AFLOW_ERROR__"
PASS = "__AWO_TESTS_PASSED__"
payload = PAYLOAD
transcript = payload["transcript"]


class NeedLLM(BaseException):
    def __init__(self, prompt, index):
        self.prompt = prompt
        self.index = index


def ensure_package(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


for package in [
    "scripts", "scripts.utils", "workspace", "workspace.HumanEval",
    "workspace.HumanEval.workflows", "workspace.HumanEval.workflows.template",
    "workspace.HumanEval.workflows.round_1",
]:
    ensure_package(package)


tree_sitter = types.ModuleType("tree_sitter")
tree_sitter.Node = type("Node", (), {})
sys.modules["tree_sitter"] = tree_sitter
sanitize_module = types.ModuleType("scripts.utils.sanitize")
exec(
    compile(payload["sanitize_source"], "scripts/utils/sanitize.py", "exec"),
    sanitize_module.__dict__,
)
sys.modules["scripts.utils.sanitize"] = sanitize_module


class FormatError(Exception):
    pass


class TextFormatter:
    def prepare_prompt(self, prompt):
        return prompt

    def validate_response(self, response):
        return True, response


class CodeFormatter:
    function_name = None

    def prepare_prompt(self, prompt):
        instructions = (
            "\n\nPlease write your code solution in Python. "
            "Return ONLY the complete, runnable code without explanations. "
            "Use proper Python syntax and formatting. "
        )
        if self.function_name:
            instructions += (
                f"\nMake sure to include a function named '{self.function_name}' in your solution. "
                "This function will be the entry point for the program."
            )
        return prompt + instructions

    def validate_response(self, response):
        python_matches = re.findall(r"```python\s*([\s\S]*?)\s*```", response)
        if python_matches:
            code = "\n\n".join(python_matches)
        else:
            generic = re.findall(r"```\s*([\s\S]*?)\s*```", response)
            code = "\n\n".join(generic) if generic else response
        try:
            code = sanitize_module.sanitize(code=code, entrypoint=self.function_name)
            if not code.strip():
                return False, None
            return True, {"response": code}
        except Exception as error:
            return False, {"error": str(error)}


class XmlFormatter:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_model(cls, model):
        return cls(model)

    def prepare_prompt(self, prompt):
        fields = getattr(self.model, "_aflow_fields", {})
        examples = [f"<{name}>{description}</{name}>" for name, description in fields.items()]
        return prompt + (
            "\n# Response format (must be strictly followed) (do not include any other "
            "formats except for the given XML format):\n" + "\n".join(examples)
        )

    def validate_response(self, response):
        matches = re.findall(r"<(\w+)>(.*?)</\1>", response, re.DOTALL)
        return True, {name: value.strip() for name, value in matches}


formatter = types.ModuleType("scripts.formatter")
formatter.BaseFormatter = object
formatter.FormatError = FormatError
formatter.XmlFormatter = XmlFormatter
formatter.CodeFormatter = CodeFormatter
formatter.TextFormatter = TextFormatter
sys.modules["scripts.formatter"] = formatter


class GenerateOp:
    _aflow_fields = {"response": "Your solution for this problem"}


class ScEnsembleOp:
    _aflow_fields = {
        "thought": "The thought of the most consistent solution.",
        "solution_letter": "The letter of most consistent solution.",
    }


class ReflectionTestOp:
    _aflow_fields = {
        "reflection_and_solution":
            "Corrective solution for code execution errors or test case failures"
    }


operator_an = types.ModuleType("workspace.HumanEval.workflows.template.operator_an")
operator_an.GenerateOp = GenerateOp
operator_an.ScEnsembleOp = ScEnsembleOp
operator_an.ReflectionTestOp = ReflectionTestOp
sys.modules[operator_an.__name__] = operator_an

op_prompt = types.ModuleType("workspace.HumanEval.workflows.template.op_prompt")
exec(compile(payload["op_prompt_source"], "op_prompt.py", "exec"), op_prompt.__dict__)
sys.modules[op_prompt.__name__] = op_prompt


class ProxyLLM:
    def __init__(self, _config=None):
        self.cursor = 0
        self.total_cost = 0.0

    async def __call__(self, prompt):
        index = self.cursor
        self.cursor += 1
        if index >= len(transcript):
            raise NeedLLM(prompt, index)
        record = transcript[index]
        if record["prompt"] != prompt:
            raise RuntimeError(f"non-deterministic relay at call {index}")
        self.total_cost += float(record.get("cost") or 0.0)
        return record["content"]

    async def call_with_format(self, prompt, formatter):
        response = await self(formatter.prepare_prompt(prompt))
        valid, parsed = formatter.validate_response(response)
        if not valid:
            raise FormatError("Response did not match the expected formatter format")
        return parsed

    def get_usage_summary(self):
        return {"total_cost": self.total_cost}


async_llm = types.ModuleType("scripts.async_llm")
async_llm.AsyncLLM = ProxyLLM
async_llm.create_llm_instance = lambda config=None: ProxyLLM(config)
sys.modules["scripts.async_llm"] = async_llm


class Logger:
    def info(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass


logs = types.ModuleType("scripts.logs")
logs.logger = Logger()
sys.modules["scripts.logs"] = logs


def extract_test_cases_from_jsonl(entry_point, dataset="HumanEval"):
    hardcoded = payload["hardcoded_tests"].get(dataset, {})
    if entry_point in hardcoded:
        return hardcoded[entry_point]
    return payload["public_tests"].get(dataset, {}).get(entry_point)


def test_case_2_test_function(solution, test_case, entry_point):
    return f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""


code_utils = types.ModuleType("scripts.utils.code")
code_utils.extract_test_cases_from_jsonl = extract_test_cases_from_jsonl
code_utils.test_case_2_test_function = test_case_2_test_function
sys.modules["scripts.utils.code"] = code_utils

operator_module = types.ModuleType("workspace.HumanEval.workflows.template.operator")
exec(compile(payload["operator_source"], "operator.py", "exec"), operator_module.__dict__)
sys.modules[operator_module.__name__] = operator_module

prompt_module_name = f"workspace.HumanEval.workflows.round_{payload['round']}.prompt"
prompt_module = types.ModuleType(prompt_module_name)
exec(compile(payload["prompt_source"], "prompt.py", "exec"), prompt_module.__dict__)
sys.modules[prompt_module_name] = prompt_module

graph_module = types.ModuleType("generated_graph")


async def execute_workflow():
    exec(compile(payload["graph_source"], "graph.py", "exec"), graph_module.__dict__)
    workflow = graph_module.Workflow(name="HumanEval", llm_config={}, dataset="HumanEval")
    value = await workflow(payload["problem"], entry_point=payload["entry_point"])
    prediction, reported_cost = value
    return {"prediction": prediction, "reported_cost": reported_cost}


try:
    output = asyncio.run(execute_workflow())
    print(RESULT + json.dumps(output, ensure_ascii=False))
except NeedLLM as request:
    print(NEED + json.dumps({"prompt": request.prompt, "index": request.index}, ensure_ascii=False))
except BaseException as error:
    print(ERROR + json.dumps({
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }, ensure_ascii=False))
print(PASS)
'''


def _marker_payload(stdout: str) -> tuple[str, dict[str, Any]]:
    for line in stdout.splitlines():
        for marker in (NEED_LLM_MARKER, RESULT_MARKER, ERROR_MARKER):
            if line.startswith(marker):
                return marker, json.loads(line[len(marker) :])
    raise RuntimeError("sandbox worker returned no AFlow relay marker")


class OfficialSandboxRuntime:
    """Execute upstream Python workflows via deterministic transcript replay."""

    def __init__(
        self,
        client: OpenRouterClient,
        sandbox: SandboxRunner,
        *,
        humaneval_public_tests: Mapping[str, Any],
        mbpp_public_tests: Mapping[str, Any] | None = None,
        max_calls: int = 40,
    ) -> None:
        self.client = client
        self.sandbox = sandbox
        self.public_tests = {
            "HumanEval": dict(humaneval_public_tests),
            "MBPP": dict(mbpp_public_tests or {}),
        }
        self.max_calls = max_calls

    @staticmethod
    def _hardcoded_tests() -> dict[str, dict[str, str]]:
        return {
            "HumanEval": {
                name: ""
                for name in (
                    "find_zero",
                    "decode_cyclic",
                    "decode_shift",
                    "by_length",
                    "add",
                    "triangle_area",
                    "correct_bracketing",
                    "solve",
                    "sum_squares",
                    "starts_one_ends",
                )
            },
            "MBPP": {
                name: ""
                for name in (
                    "remove_odd",
                    "replace_spaces",
                    "snake_to_camel",
                    "Split",
                    "swap_List",
                    "square_Sum",
                    "sort_sublists",
                    "unique_sublists",
                )
            },
        }

    def _worker_source(
        self,
        candidate: OfficialPythonCandidate,
        *,
        round_number: int,
        problem: str,
        entry_point: str,
        transcript: list[dict[str, Any]],
    ) -> str:
        values = _literal_assignments("scripts/prompts/optimize_prompt.py")
        graph_source = values["WORKFLOW_TEMPLATE"].format(
            graph=candidate.graph,
            round=round_number,
            dataset="HumanEval",
        )
        payload = {
            "round": round_number,
            "problem": problem,
            "entry_point": entry_point,
            "graph_source": graph_source,
            "prompt_source": candidate.prompt,
            "operator_source": _source("workspace/HumanEval/workflows/template/operator.py"),
            "op_prompt_source": _source("workspace/HumanEval/workflows/template/op_prompt.py"),
            "sanitize_source": _source("scripts/utils/sanitize.py"),
            "public_tests": self.public_tests,
            "hardcoded_tests": self._hardcoded_tests(),
            "transcript": transcript,
        }
        return "PAYLOAD = " + repr(payload) + "\n" + textwrap.dedent(_WORKER)

    def execute(
        self,
        candidate: OfficialPythonCandidate,
        *,
        round_number: int,
        problem: str,
        entry_point: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> OfficialExecution:
        transcript: list[dict[str, Any]] = []
        sandbox_runs = 0
        while sandbox_runs <= self.max_calls:
            sandbox_runs += 1
            source = self._worker_source(
                candidate,
                round_number=round_number,
                problem=problem,
                entry_point=entry_point,
                transcript=transcript,
            )
            result = self.sandbox.run(source)
            if not result.passed:
                raise RuntimeError(
                    f"generated workflow sandbox failed: {result.status}: {result.stderr[-1000:]}"
                )
            marker, payload = _marker_payload(result.stdout)
            if marker == ERROR_MARKER:
                raise RuntimeError(
                    f"generated workflow raised {payload.get('type')}: {payload.get('message')}"
                )
            if marker == RESULT_MARKER:
                return OfficialExecution(
                    prediction=str(payload["prediction"]),
                    reported_cost=float(payload.get("reported_cost") or 0.0),
                    relay_cost=sum(float(item.get("cost") or 0.0) for item in transcript),
                    tokens=sum(int(item.get("tokens") or 0) for item in transcript),
                    calls=len(transcript),
                    sandbox_runs=sandbox_runs,
                )
            if len(transcript) >= self.max_calls:
                raise RuntimeError(f"generated workflow exceeded {self.max_calls} LLM calls")
            index = int(payload["index"])
            if index != len(transcript):
                raise RuntimeError(f"unexpected relay index {index}")
            prompt = str(payload["prompt"])
            chat = self.client.chat(
                [{"role": "user", "content": prompt}],
                metadata={
                    **dict(metadata or {}),
                    "role": "aflow_executor",
                    "dataset": "HumanEval",
                    "round": round_number,
                    "workflow_call": index,
                    "upstream_commit": PAPER_COMMIT,
                },
            )
            transcript.append(
                {
                    "prompt": prompt,
                    "content": chat.content,
                    "cost": chat.usage.cost,
                    "tokens": chat.usage.total_tokens,
                }
            )
        raise AssertionError("workflow relay loop exited unexpectedly")
