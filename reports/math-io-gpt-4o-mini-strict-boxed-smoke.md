# GPT-4o-mini MATH IO strict-boxed-v2 smoke

运行日期：2026-09-04。

`controlled/strict-boxed-v2` 去掉 JSON 包装，直接要求模型只输出完整的
`\\boxed{<answer>}`。prompt 明确要求符号、单位、货币、集合、矩阵和变量均置于 boxed
内部，并禁止代码、解释、推导和占位符。实现提交为
`ff2cf91ebc423f8b116aa3f1f7760b00dd5040fe`。

```bash
python scripts/run_strict_math_io_experiment.py \
  data/raw/datasets/math_test.jsonl \
  experiments/runs/math-io-gpt-4o-mini-strict-boxed/smoke-30 \
  --repeats 1 --start 0 --limit 30 \
  --config configs/math_io_gpt_4o_mini_strict_boxed.yaml
```

验收结果：

- 记录和实际模型：30/30，均为 `openai/gpt-4o-mini`；失败 0。
- 原始响应只有 boxed：19/30。
- 原始响应包含可由 MATH scorer 提取的 boxed 最终答案：30/30。
- 盒外推导：11/30；代码式输出：0/30；占位符残留：0/30。
- 正确数：9/30（30%），仅作为链路 smoke，不作为正式指标。
- token：12,836；费用：$0.00491520。

模型偶尔无视“只输出 boxed”的要求而给出推导，但 archive MATH scorer 固定提取最后一个
boxed 答案，因此盒外文本不改变判分。相比 JSON v1，这一协议不再引入 LaTeX 反斜杠转义、
尾随 JSON 括号或 boxed 外单位等包装问题。正式实验使用独立目录，不复用 v1 或 smoke 记录。
