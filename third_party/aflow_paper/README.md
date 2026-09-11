# AFlow paper source snapshot

This directory contains the minimal HumanEval optimization sources copied verbatim from:

- repository: `https://github.com/FoundationAgents/AFlow.git`
- commit: `289d5544cfb5698799b207b931a392aadc9e9c1a`
- license: MIT (see `LICENSE`)

The snapshot is intentionally limited to the optimizer, its helper modules, the HumanEval
benchmark and initial workflow/operator templates. AWO compatibility and sandbox code lives
outside this directory so that these files remain byte-identical to upstream.

To verify a file against a fresh upstream checkout:

```bash
git clone https://github.com/FoundationAgents/AFlow.git /tmp/AFlow
git -C /tmp/AFlow checkout 289d5544cfb5698799b207b931a392aadc9e9c1a
cmp third_party/aflow_paper/scripts/prompts/optimize_prompt.py \
  /tmp/AFlow/scripts/prompts/optimize_prompt.py
```
