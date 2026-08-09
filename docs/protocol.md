# Table 1 复现协议

## 1. 协议版本

- 协议版本：`1.0.0`
- 冻结日期：`2026-08-09`
- 论文版本：AFlow arXiv v4 / ICLR 2025
- 主实验：论文 Table 1
- 模型替换：所有 LLM 角色统一为 OpenRouter `deepseek/deepseek-chat`

任何影响样本、prompt、模型参数、调用图、搜索预算或评分规则的改动，都必须提升协议版本，并在报告中与旧版本分开。

## 2. 实验矩阵

数据集：

- HotpotQA
- DROP
- HumanEval
- MBPP
- GSM8K
- MATH Level 5

方法：

- IO
- CoT
- CoT-SC
- MedPrompt
- MultiPersona
- Self-Refine
- ADAS
- AFlow

每个冻结后的方法/工作流在测试集独立执行 3 次。三个重复不得跨 run 复用 LLM 响应。

## 3. 统一模型设置

所有角色，包括 AFlow optimizer、AFlow operators、ADAS meta-agent、ADAS executor、baseline generator、reviewer、reviser、voter 和 selector，使用：

| 参数 | 值 |
| --- | --- |
| Provider | OpenRouter |
| Model | `deepseek/deepseek-chat` |
| Base URL | `https://openrouter.ai/api/v1` |
| Temperature | `1.0` |
| Candidates per request | `n=1` |
| API key | 环境变量 `OPENROUTER_API_KEY` |

多候选方法必须发起多次独立的 `n=1` 请求。每次请求记录请求模型、API 返回模型、路由 provider、采样参数、token、延迟、费用、请求 ID、prompt hash 和错误。

OpenRouter 模型名是别名而不是论文快照，因此该实验属于统一模型的协议复现，不属于论文 Claude/GPT-4o-mini 组合的 exact model reproduction。

## 4. 数据

使用 `data/manifests/aflow-artifacts.json` 锁定的 AFlow 官方数据档案。

| 数据集 | Validation | Test | 主指标 |
| --- | ---: | ---: | --- |
| HotpotQA | 200 | 800 | token F1 |
| DROP | 200 | 800 | token F1 |
| HumanEval | 33 | 131 | pass@1 |
| MBPP | 86 | 341 | pass@1 |
| GSM8K | 264 | 1,055 | solve rate |
| MATH | 119 | 486 | solve rate |

`*_public_test.jsonl` 仅用于兼容上游公开测试格式，不属于 Table 1 测试分母。禁止重新随机切分官方工件。

## 5. 评测

### 5.1 QA

HotpotQA 和 DROP 使用同一答案规范化与 token F1 实现：

1. 小写化；
2. 去标点；
3. 去冠词；
4. 合并空白；
5. 计算 token precision/recall/F1；
6. 多 gold answer 取最大值。

### 5.2 数学

- GSM8K：从约定答案字段或最后数值提取答案，兼容逗号、符号、小数和科学计数法。
- MATH：优先取最后一个 `\boxed{}`，随后执行规范化和符号等价判断。
- 解析失败记 0 分，不使用 LLM 裁判。

### 5.3 代码

- HumanEval 和 MBPP 报告 pass@1。
- 生成代码必须在隔离沙箱中运行。
- 编译错误、运行错误、超时和沙箱违规记 0 分。
- 不因执行错误改变原始分母。

## 6. Baseline 调用预算

| 方法 | 单样本协议 |
| --- | --- |
| IO | 1 次直接生成 |
| CoT | 1 次 step-by-step 生成 |
| CoT-SC | 5 次候选生成 + 1 次 selector |
| MedPrompt | 3 次候选生成 + 5 次顺序扰动投票 |
| MultiPersona | 3 个 persona × 2 轮 + 1 次汇总 |
| Self-Refine | 初始生成 + 最多 3 轮 review/revise |

Self-Refine 提前停止必须保存停止原因。MedPrompt 必须保存 shuffle seed。多候选方法必须保存每个候选，不能只保存最终答案。

## 7. AFlow

### 7.1 搜索设置

- 最大轮数：20
- top-k：3
- early-stop：top-k 连续 5 轮不变
- 每个新 workflow 的 validation 重复：5
- 主选择参数：`alpha=0.4`、`lambda=0.2`
- 搜索数据：仅 validation
- 测试：冻结最佳 workflow 后独立运行 3 次

另外运行两组敏感性配置：

- 附录：`alpha=0.2`、`lambda=0.4`
- 当前独立仓库：`alpha=0.2`、`lambda=0.3`

三组结果分开保存，主表只使用正文配置。

### 7.2 Operators

| 数据集 | Operators |
| --- | --- |
| HotpotQA、DROP | Custom、AnswerGenerate、ScEnsemble |
| GSM8K、MATH | Custom、ScEnsemble、Programmer |
| HumanEval、MBPP | Custom、CustomCodeGenerate、ScEnsemble、Test |

## 8. ADAS

- 算法：Meta Agent Search
- 搜索轮数：30
- optimizer/executor：OpenRouter `deepseek/deepseek-chat`
- 搜索数据：仅 validation
- 测试：冻结最佳 agent 后独立运行 3 次

六任务 adapter 为本项目实现，必须标记为 `protocol-compatible`。

## 9. 重试、缓存和失败

- 重试上限：8 次；
- 对 429、5xx 和瞬时网络错误执行带 jitter 的指数退避；
- 相同 run 内可用请求 hash 恢复中断任务；
- 三个正式重复之间不得复用响应；
- 重试不得改变 prompt 或采样参数；
- 最终失败样本记 0 分并进入错误报告。

## 10. 统计与报告

每个 dataset/method 报告：

- 三次原始分数；
- mean、standard deviation、95% CI；
- 论文值及绝对差；
- 成功数、API 失败数、解析失败数、执行错误数和超时数；
- 请求数、输入/输出 token、延迟和实际费用；
- AFlow/ADAS 搜索成本与冻结工作流测试成本；
- 复现等级：`artifact-replay`、`paper-faithful`、`archive-faithful`、`corrected-controlled`、`protocol-compatible` 或 `unresolved`。

与论文值相差超过 2 个百分点时做样本级诊断，但不得使用测试集调 prompt 或筛选结果。
