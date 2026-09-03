# GPT-4o-mini：MATH IO 对照实验

运行日期：2026-09-03。

## 实验口径

- 数据：`data/raw/datasets/math_test.jsonl`，完整 486 条 test 样本。
- 方法：IO（`paper-faithful` baseline protocol）。
- 模型：OpenRouter `openai/gpt-4o-mini`。
- 重复：3 次独立请求，不跨 repeat 复用响应。
- 温度：1.0；每次请求最多 4096 output tokens；`n=1`。
- 注册配置：`configs/math_io_gpt_4o_mini.yaml`。
- 实现提交：`df6a9c657c42b90062bf5f9c4a81c6a9b1c56c11`。
- 输出：`experiments/runs/math-io-gpt-4o-mini/math/test/shards/`。

为缩短串行运行时间，486 条样本被划分为 8 个互不重叠的分片。`--start/--limit`
分别为 `0/61`、`61/61`、`122/61`、`183/61`、`244/61`、`305/61`、
`366/60` 和 `426/60`；每个分片均执行 `--methods io --repeats 3`。例如：

```bash
python scripts/run_manual_experiment.py \
  math data/raw/datasets/math_test.jsonl \
  experiments/runs/math-io-gpt-4o-mini/math/test/shards/000-060 \
  --methods io --repeats 3 --start 0 --limit 61 \
  --config configs/math_io_gpt_4o_mini.yaml
```

## 正式结果

| Repeat | 正确数 | 样本数 | Accuracy | Tokens | Cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 57 | 486 | 11.7284% | 235,254 | $0.10351395 |
| 2 | 55 | 486 | 11.3169% | 236,090 | $0.10391955 |
| 3 | 52 | 486 | 10.6996% | 227,142 | $0.09855075 |
| 平均/合计 | — | 1,458 | **11.2483%** | **698,486** | **$0.30598425** |

完整性检查：1,458 个预期 key 全部存在，缺失 0、重复 0、意外 key 0、运行失败 0。
1,458 次请求均在第一次 attempt 成功并以 `stop` 结束；requested model 与 actual model
均为 `openai/gpt-4o-mini`。OpenRouter 实际路由为 OpenAI 744 次、Azure 714 次。

正式平均准确率比同一仓库中 DeepSeek Chat 的 MATH IO 均值 5.0069% 高 6.2414 个百分点。
两者都明显受当前 IO 输出协议影响，因此这个差值只表示当前复现配置下的对照结果，不代表
模型的一般 MATH 能力。

## 输出格式诊断

当前所有数据集共享的 IO system prompt 包含：`For code tasks, put the complete Python
program in that field.`。虽然 MATH user prompt 要求最终答案使用 `\\boxed{...}`，
GPT-4o-mini 在 1,458 个预测中仍有 983 个（67.42%）被启发式识别为代码式回答；其中仅
8 个判对。共有 1,202 个预测包含 `\\boxed`，但不少代码式回答把运行时变量而非已求出的
常量放入 boxed 表达式。

这说明 11.2483% 是“当前 IO 提示词 + JSON 解析器 + archive MATH scorer”的有效正式
分数，但不能直接解释成 GPT-4o-mini 的标准 MATH benchmark 能力。若要测试纯数学 IO，
应先把按数据集生成的 system schema 与 code-only 指令解耦，注册为新的 protocol/config，
再另行运行；不能用修改后的提示词覆盖本次结果。

## 排除的串行试跑

最初的单进程串行试跑在切换到分片前被停止，其 23 个已落盘请求保存在
`experiments/runs/math-io-gpt-4o-mini/math/test/aborted-serial-20260903/`，不参与正式分数。
这些请求记录了 8,970 tokens、$0.00326565。正式实验加该试跑的本地可核对费用为
$0.30924990；被中断时还有一个在途请求，它可能已被上游计费但没有进入本地账本。
