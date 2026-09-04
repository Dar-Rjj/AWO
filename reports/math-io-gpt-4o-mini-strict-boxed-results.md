# GPT-4o-mini MATH IO strict-boxed-v2 正式结果

运行日期：2026-09-04。

## 输出协议

本次正式结果使用 `controlled/strict-boxed-v2`。实际 system prompt 为：

```text
Return only the final answer in the exact form \boxed{<answer>} and nothing else.
Replace <answer> with the complete final mathematical answer. Keep every required
sign, unit, currency symbol, tuple/list/set element, matrix expression, and symbolic
variable inside the braces. Do not output JSON, Markdown fences, prose, a derivation,
or an unevaluated placeholder.
```

该协议移除了原通用 IO prompt 的代码任务语句，也移除了 JSON 包装及其 LaTeX 反斜杠
转义问题。模型仍可能不服从“只输出最终答案”而在 boxed 外给出推导；MATH scorer 固定
提取最后一个完整 boxed，因此盒外文本不会进入数学等价判断。

## 实验口径

- 数据：完整 `data/raw/datasets/math_test.jsonl`，486 条 Level 5 test 样本。
- 模型：OpenRouter `openai/gpt-4o-mini`。
- 方法：单请求 IO；prompt 不要求 CoT；temperature 1.0；`n=1`。
- 重复：3 次独立请求，不跨 repeat 复用。
- 配置：`configs/math_io_gpt_4o_mini_strict_boxed.yaml`。
- protocol：`controlled/strict-boxed-v2`。
- config SHA256：`e9862225afa0634b8092b9c698425839102f29ada1744d57ce23cfe920d1f881`。
- manifest implementation commit：`8f8f91d586f0b0a226cfe20a368129158cb3cbe5`。
- 输出：`experiments/runs/math-io-gpt-4o-mini-strict-boxed/math/test/shards/`。

486 条样本按 `--start/--limit` 分为 `0/61`、`61/61`、`122/61`、`183/61`、
`244/61`、`305/61`、`366/60`、`426/60` 八个互不重叠的并行分片，每个分片执行
三次 repeat。

## 正式结果

| Repeat | 正确数 | 样本数 | Accuracy | Tokens | Cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 167 | 486 | 34.3621% | 290,306 | $0.13348335 |
| 2 | 164 | 486 | 33.7449% | 292,798 | $0.13488255 |
| 3 | 169 | 486 | 34.7737% | 287,297 | $0.13158195 |
| 平均/合计 | — | 1,458 | **34.2936%** | **870,401** | **$0.39994785** |

完整性与模型审计：

- 1,458/1,458 个预期 key 存在；缺失 0、重复 0、意外 key 0、流程失败 0。
- 1,458 次调用均为第一次 attempt 成功，finish reason 均为 `stop`。
- requested/actual model 1,458/1,458 均为 `openai/gpt-4o-mini`。
- OpenRouter 路由：Azure 738 次、OpenAI 720 次。
- protocol 标签 1,458/1,458 均为 `controlled/strict-boxed-v2`。

## 格式质量与假阴性审计

- 原始响应只有一个 boxed：550/1,458（37.72%）。
- 含至少一个完整、可提取 boxed：1,456/1,458（99.86%）。
- boxed 外存在文本：906/1,458（62.14%）。
- 代码式输出：0；`<answer>` 占位符残留：0。
- 500 个正确结果中，70 个是纯 boxed，430 个含盒外文本；scorer 只使用最后一个 boxed。

仅 `math-test-0331` 的 repeat 1 和 2 完全没有 boxed。人工核对原题、参考答案和原始响应：
参考答案为 `3R^2`，两次响应都错误地加入了额外边长项，其中一次还使用了错误系数。因此它们
不是“数学答案正确、只因格式漏盒而判 0”。本轮没有发现输出包装造成的假阴性。

“prompt 不影响模型本身的求解行为”无法从实验设计上绝对保证；这里能够验证的是：代码指令、
JSON 转义和盒外文本不再污染最终数学答案的提取与评分。模型自发输出推导的行为作为格式合规
指标单独报告，不静默当作纯 boxed 输出。

## 与旧 prompt 的对照

| 协议 | MATH IO mean | 代码式输出 | 正式费用 |
| --- | ---: | ---: | ---: |
| 通用 JSON + code-task 语句 | 11.2483% | 983/1,458 | $0.30598425 |
| strict JSON v1（中间排障，不作为最终结果） | 9.3278% | 0/1,458 | $0.06578955 |
| **strict-boxed-v2（最终）** | **34.2936%** | **0/1,458** | **$0.39994785** |

v2 比最初受代码指令污染的结果高 23.0453 个百分点。由于 prompt 协议发生变化，这不是同一
实验记录的覆盖更新；三套原始账本均保留，Table 1 主实验也不会自动用 v2 替换已冻结的 IO
结果，除非后续明确注册统一的新版本协议。

本次修正过程额外执行了三次 30 条 smoke 和 strict JSON v1 全量排障。v1 smoke
$0.00109230、修正版 v1 smoke $0.00108150、v2 smoke $0.00491520、v1 全量
$0.06578955，加上最终 v2 全量 $0.39994785，本轮本地可核对总费用为 **$0.47282640**。
