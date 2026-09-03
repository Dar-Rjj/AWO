# Table 1 IO/CoT full-test status

Date: 2026-09-03

## Protocol

- Model: OpenRouter `deepseek/deepseek-chat` for every inference call.
- Scope: six complete test splits, IO and CoT, three independent repeats.
- Search/training: none.
- Registered config: `configs/paper/table1.yaml`.
- Persisted implementation commit: `8bf8283cf5ac5be6e6547ad875af8d4591b857d0`.

## Request audit

- Expected and completed records: 21,678 / 21,678.
- Successful model calls represented by final records: 21,678.
- Prompt tokens: included in the append-only request ledgers.
- Final-record total tokens: 13,130,026.
- Final-record cost: $5.880010.
- Initial provider responses with empty `choices`: 25. Their failed records were moved to
  `retry-archive/empty-choices-20260903` and every missing key was successfully rerun.
- An abandoned mixed-directory launch issued 255 additional successful calls costing $0.048573;
  it is isolated under `io-cot-aborted-20260903` and excluded from reported scores.
- Total API spend attributable to this stage: approximately $5.928583.

## Validated scores

Scores are percentages averaged over three repeats. HumanEval and MBPP remain intentionally
withheld because this host cannot create the required Docker sandbox.

| Dataset | IO | CoT | Status |
| --- | ---: | ---: | --- |
| DROP | 82.6513 | 20.6635 | valid |
| GSM8K | 29.4787 | 92.7962 | valid |
| HotpotQA | 73.6653 | 72.2229 | valid |
| MATH | 5.0069 | 38.3402 | valid |
| HumanEval | — | — | 786 predictions await Docker rescoring |
| MBPP | — | — | 2,046 predictions await Docker rescoring |

## Code-evaluation blocker and recovery

All 2,832 HumanEval/MBPP final records contain successful model predictions, but their current
score details are `sandbox_error` because `docker` was absent during the run. Docker 29.1.3 was
installed afterward. The daemon can start with networking disabled, but the host denies the
namespace operation required to build or start containers (`unshare: operation not permitted`).
Running generated code outside the registered sandbox is not an acceptable fallback.

`scripts/rescore_manual_code.py` now rescales no API traffic: it reads the persisted prediction,
runs the existing benchmark harness in `DockerSandbox`, atomically replaces only records whose
prior status was `sandbox_error`, recomputes the integrity wrapper, and regenerates `summary.json`.
On a Docker-capable host, run it once for each of these four directories:

```bash
python scripts/rescore_manual_code.py humaneval data/raw/datasets/humaneval_test.jsonl \
  experiments/runs/aflow-table1-deepseek-chat/humaneval/test/io
python scripts/rescore_manual_code.py humaneval data/raw/datasets/humaneval_test.jsonl \
  experiments/runs/aflow-table1-deepseek-chat/humaneval/test/cot
python scripts/rescore_manual_code.py mbpp data/raw/datasets/mbpp_test.jsonl \
  experiments/runs/aflow-table1-deepseek-chat/mbpp/test/io
python scripts/rescore_manual_code.py mbpp data/raw/datasets/mbpp_test.jsonl \
  experiments/runs/aflow-table1-deepseek-chat/mbpp/test/cot
```

The rescorer has unit coverage. Repository verification after the change: 190 tests passed,
6 Docker-only security tests skipped, and Ruff passed.
