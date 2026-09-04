# GPT-4o-mini MATH IO strict-output smoke

运行日期：2026-09-04。

## 目的

在正式重跑前验证 `controlled/strict-output-v1` 已消除通用 IO system prompt 中的代码任务
指令，并且不会禁止 MATH 合法的含变量符号表达式。

最终 prompt 明确要求：

- 整个响应只能是一个 JSON 对象；
- 唯一字段为字符串 `solution`；
- 字段值为 `\\boxed{<answer>}`；
- `<answer>` 必须替换成最终数学数值或表达式；
- 不允许保留占位符或待执行的计算指令；
- prompt 中不出现 `Python`、`program` 或 `code`。

实现提交：`26764ca985178b5136cf9ef861cb4b16295b7b56`。

## 最终 smoke 结果

命令：

```bash
python scripts/run_strict_math_io_experiment.py \
  data/raw/datasets/math_test.jsonl \
  experiments/runs/math-io-gpt-4o-mini-strict/smoke-30-symbolic \
  --repeats 1 --start 0 --limit 30 \
  --config configs/math_io_gpt_4o_mini_strict.yaml
```

- 记录：30/30；API/流程失败：0。
- 实际模型：30/30 `openai/gpt-4o-mini`。
- protocol 标签：30/30 `controlled/strict-output-v1`。
- 完整合法 JSON 且仅含 `solution`：29/30。
- 解析后的纯 boxed 最终答案：30/30。
- 代码式输出：0/30。
- `<answer>` 占位符残留：0/30。
- token：6,289；费用：$0.00108150。
- 小样本正确数：5/30（16.67%）；这里只作为链路检查，不作为正式指标。

唯一非严格 JSON 响应只多了一个尾随 `}`。既有容错 JSON-object parser 仍完整取出其
`solution` 字符串，解析后的 prediction 是纯 boxed 表达式，所以没有发生由输出包装导致的
误判。正式实验必须使用新输出目录，不复用任何旧 prompt 的记录。
