# 来源与可追溯性

本文件记录阶段 0 已核验的一手来源。机器可读的 commit 和 SHA256 位于：

- `third_party/upstream.lock.yaml`
- `data/manifests/aflow-artifacts.json`

## 1. 论文

- 标题：AFlow: Automating Agentic Workflow Generation
- arXiv：https://arxiv.org/abs/2410.10762
- 固定版本：https://arxiv.org/pdf/2410.10762v4
- 最后修订：2025-04-15
- PDF SHA256：`9be15f695f11dd5bc634c1c026bd2270eff3d3c4a53c4d9b51c012b7bd03d521`

## 2. AFlow

- 仓库：https://github.com/FoundationAgents/AFlow
- 论文修订日附近提交：`289d5544cfb5698799b207b931a392aadc9e9c1a`
- 提交时间：2025-04-15T17:52:02+08:00
- 提交说明：`Fix HumanEval Bug.`
- 核验时 main：`3f457218fc716093fe53f6df8a5d5e6379d66346`
- 许可证：MIT

论文同期提交用于 paper-faithful 移植；当前 main 仅用于缺陷对照和兼容性参考。当前仓库 README 明确提示部分 operators 在从 MetaGPT 迁移时可能存在 bug。

## 3. 历史 baselines

- 仓库：https://github.com/FoundationAgents/MetaGPT
- 固定提交：`22e8f9d7fcfabd4c1fc6640235079c1c70bc5899`
- 提交时间：2024-09-22T15:46:50+08:00
- 提交说明：`Update baseline and benchmark; update evaluator`
- 路径：`examples/ags/experiments/baselines/`
- 文件数：29
- 许可证：MIT

公开代码覆盖：

| 方法 | DROP | GSM8K | HotpotQA | HumanEval | MATH | MBPP |
| --- | --- | --- | --- | --- | --- | --- |
| IO | 缺 | 有 | 有 | 有 | 有 | 有 |
| CoT | 有* | 有 | 有 | 有 | 有 | 有 |
| CoT-SC | 缺 | 有 | 有 | 有 | 有 | 有 |
| MedPrompt | 缺 | 有 | 缺 | 有 | 有 | 有 |
| MultiPersona | 缺 | 有 | 有 | 有 | 有 | 有 |
| Self-Refine | 缺 | 有 | 缺 | 有 | 有 | 有 |

`cot_drop.py` 虽存在，但为明显的临时/异常配置，不能直接视为 Table 1 完整实现。缺失组合的 task adapter 和 prompt 必须在来源文档中标记为 inferred。

## 4. ADAS

- 仓库：https://github.com/ShengranHu/ADAS
- 固定提交：`2702bee8fefda42255efc5be9f60e3bd3db96ae4`
- 提交时间：2025-01-27T22:37:51-08:00
- 提交说明：`fix role description typo`
- 许可证：Apache-2.0

ADAS 上游公开了多个自包含 domain，但没有 AFlow Table 1 所用的六任务统一 adapter、初始 archive 和完整搜索配置。本项目将复用 Meta Agent Search，并将自建部分标记为 `protocol-compatible`。

## 5. OpenRouter 客户端参考

本地参考：

- `/home/rjj/RealEvo/utils/llm_client/openrouter.py`
- `/home/rjj/RealEvo/utils/llm_client/openai.py`
- `/home/rjj/RealEvo/utils/llm_client/base.py`
- `/home/rjj/RealEvo/cfg/llm_client/openrouter.yaml`

已核验行为：

- 环境变量：`OPENROUTER_API_KEY`
- Base URL：`https://openrouter.ai/api/v1`
- 模型：`deepseek/deepseek-chat`
- 默认温度：`1.0`
- API：OpenAI-compatible `chat.completions.create`
- OpenRouter 子类单次只接受 `n=1`

本项目会保留接口语义，但将参考实现的最多 1,000 次重试改为有限重试，并增加请求审计。

## 6. 官方工件

AFlow `data/download_data.py` 在固定提交中给出三个 Google Drive 工件：

| 工件 | Google Drive ID | SHA256 |
| --- | --- | --- |
| 数据 | `1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e` | `7089a373c27184ae1b67e322f8276642dab246359052989b03a27f48251ae053` |
| 结果 | `1Sr5wjgKf3bN8OC7G6cO3ynzJqD4w6_Dv` | `f460f9c9218216f6fbaf553f4f215248317b7f37dd715ae146cd38199eabe5f7` |
| 初始轮次 | `1UBoW4WBWjX2gs4I_jq3ALdXeLdwDJMdP` | `eeb0a1ddf7b915e3405d35e5952db2b17fb86f5c01f1988e7abb4a721ef76cab` |

阶段 0 已实际下载、识别文件类型、成功列出/解包并计算哈希。大型 Google Drive 结果不能依赖简单 `curl -L`，下载器需要处理确认页面，并在解包前验证 MIME 与 SHA256。

## 7. 代码评测镜像

- 基础镜像：`python:3.9-slim`，固定 repo digest
  `sha256:2d97f6910b16bd338d3060f261f53f144965f755599aab1acda1e13cf1731b1b`。
- Python：3.9.25。
- NumPy：2.0.2，CPython 3.9 manylinux2014 x86_64 wheel SHA256
  `f26b258c385842546006213344c50655ff1555a9338e2e5e02a0756dc3e803dd`；同一哈希也存在于
  `requirements.lock`。
- 构建定义：`docker/code-sandbox/Dockerfile`；运行时无网络、无宿主挂载。
- 本机验证 Image ID：
  `sha256:3ac37a78e2e380ddb53b9c0ca9b672ffde319605113d97620fb92b38ff022861`。

## 8. 来源优先级

冲突时按以下优先级处理，并保留敏感性配置：

1. 论文正文明确值；
2. 论文附录；
3. 论文同期公开代码；
4. 官方发布数据和结果工件；
5. 当前独立仓库；
6. 原始 baseline/ADAS 论文及仓库；
7. 本项目的最小推断。

任何第 7 类内容都不能标记为 exact。
