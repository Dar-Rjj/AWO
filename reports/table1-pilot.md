# Table 1 pilot report

This report is the paid-run gate immediately before the full Table 1 reproduction. It does not
contain full Table 1 results and does not authorize the full run.

## Provenance and completion

- Implementation commit: `14e0acaf6b20a5c70ee49d0fab47a6c1ffdb6312`
- Config: `configs/pilot.yaml`
- Config SHA256: `8189674cd03a467c1bbb39b90b19fa2612761bc41c46b14e8ce76c2e32893076`
- Model for every role: `deepseek/deepseek-chat` through OpenRouter
- Scope: 6 datasets, 8 methods, 20 validation/test examples per dataset, 1 test repeat
- Completion: 30/30 jobs, 12/12 frozen search candidates, 48/48 result cells, 960/960
  test sample-method records
- Resume check: replaying the complete command kept the request ledger at
  `12,450 -> 12,450`; no OpenRouter request was added.

The generated machine-readable report is kept locally under the intentionally ignored
`reports/generated/table1-pilot/` directory. Experiment ledgers are under the intentionally
ignored `experiments/runs/aflow-table1-pilot/` directory.

## Pilot scores

Scores are percentages over a bounded sample of 20 and are only a pipeline and budget signal;
they are not estimates with enough power to replace the full experiment.

| Dataset | IO | CoT | CoT-SC | MedPrompt | MultiPersona | Self-Refine | ADAS | AFlow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 75.45 | 70.87 | 75.56 | 76.55 | 73.33 | 68.79 | 72.48 | 73.33 |
| DROP | 76.67 | 27.40 | 36.81 | 15.32 | 80.00 | 18.31 | 68.50 | 83.33 |
| HumanEval | 90.00 | 90.00 | 75.00 | 30.00 | 90.00 | 85.00 | 90.00 | 90.00 |
| MBPP | 85.00 | 80.00 | 80.00 | 55.00 | 85.00 | 85.00 | 80.00 | 80.00 |
| GSM8K | 25.00 | 90.00 | 90.00 | 90.00 | 95.00 | 85.00 | 80.00 | 95.00 |
| MATH | 5.00 | 45.00 | 35.00 | 50.00 | 70.00 | 55.00 | 35.00 | 65.00 |

## Actual usage and audit

| Phase | Requests | Tokens | Cost (USD) | Successful terminal records |
| --- | ---: | ---: | ---: | ---: |
| Validation search | 8,311 | 6,254,096 | $2.985265 | 8,305 |
| Frozen test | 4,139 | 3,248,458 | $1.490875 | 4,139 |
| Total | 12,450 | 9,502,554 | $4.476140 | 12,444 |

All 4,139 test requests used requested/actual model `deepseek/deepseek-chat`. The 8,305
successful search requests did too. OpenRouter routed successful requests through DeepInfra,
Novita, and StreamLake. The six terminal search-request errors all occurred in the interrupted
MATH AFlow round: three upstream 429 responses and three responses with no choices. Search
evaluation scored these fail-closed and completed. The ledger also includes the partial MATH
round that had to be resumed, so the amount above is actual spend, not an idealized clean-run
cost.

Test evaluation recorded 44 fail-closed sample-method outcomes across the 960 records. They
remain in the denominator with score zero, as fixed by the protocol; no failed sample was
dropped or regenerated selectively.

## Full-run budget estimate

The full dry-run plan selects 3,613 test examples per repeat, 3 repeats, all validation examples,
30 ADAS rounds, and 20 AFlow generated rounds with 5 validation repeats. Its implementation-level
logical-call bound, excluding transport retries, is 421,397 to 1,827,506 requests.

Two estimates are retained because workflow depth and prompt length evolve during search:

| Estimate | Requests | Tokens | Cost (USD) |
| --- | ---: | ---: | ---: |
| Pilot blended rate applied to static bounds | 421,397-1,827,506 | 322M-1.395B | $151.50-$657.04 |
| Per-dataset pilot-rate point estimate | 807,134 | 649M | $302.75 |

The point estimate scales every test cell by its dataset's full test size and three repeats. ADAS
search is scaled by validation size and `(7+30)/(7+3)` initial-plus-generated architectures. AFlow search is scaled by validation size,
`21/4` evaluated candidates, and `5/2` validation repeats. It assumes no early stop and preserves
pilot-average workflow complexity. Its cost breakdown is approximately $164.20 for search
($52.33 ADAS and $111.86 AFlow) plus $138.55 for frozen tests.

These are planning estimates, not a hard cap: later workflows can have longer contexts, provider
pricing/routing can change, and retries can add requests. A practical authorization should cover
at least the conservative `$657.04` estimate plus an explicit contingency, or run the full matrix
in staged dataset checkpoints. Until the user explicitly approves the resulting budget, the full
command remains protected by `--confirm-full-table1` and must not be started.
