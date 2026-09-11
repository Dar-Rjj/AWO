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

### 3.1 IO prompt

固定 commit 中的 IO user prompt 来源：

| 数据集 | 历史文件 | 本项目标记 |
| --- | --- | --- |
| HotpotQA | `io_hotpotqa.py` | `upstream-user/adapted-schema` |
| DROP | 无 | `paper-faithful/inferred` |
| GSM8K | `io_gsm8k.py` | `upstream-user/adapted-schema` |
| MATH | `io_math.py` | `upstream-user/adapted-schema` |
| HumanEval | `io_humaneval.py` | `upstream-user/adapted-schema` |
| MBPP | `io_mbpp.py` | `upstream-user/adapted-schema` |

可审计文本位于 `prompts/baselines/io/`，运行时模板位于 `src/awo/baselines/io.py`，测试会
逐字检查两者一致。`adapted-schema` 表示保留历史 user prompt，但用显式 JSON system 消息
替代 MetaGPT `ActionNode` 的 Pydantic 填充；该兼容层不产生额外 LLM 请求。

### 3.2 CoT prompt

| 数据集 | 历史文件 | 本项目标记 |
| --- | --- | --- |
| HotpotQA | `cot_hotpotqa.py` | `upstream-user/adapted-schema` |
| DROP | `cot_drop.py`（入口/参数不兼容） | `paper-faithful/inferred` |
| GSM8K | `cot_gsm8k.py` 的 GPT 变体 | `upstream-user/adapted-schema` |
| MATH | `cot_math.py` | `upstream-user/adapted-schema` |
| HumanEval | `cot_humaneval.py` | `upstream-user/adapted-schema` |
| MBPP | `cot_mbpp.py` | `upstream-user/adapted-schema` |

可审计文本位于 `prompts/baselines/cot/`。GSM8K 固定使用历史代码实际选择的
`GSM8K_PROMPT_GPT`，而非文件中未调用的 DeepSeek 变体，以保持 Table 1 执行路径。
DROP 历史入口将只含 `context` 的 evaluator 连接到要求 `question, context` 两个参数的 graph，
且默认 `samples=1`；主协议未将其误标为可运行的论文实现。

### 3.3 CoT-SC prompt 与调用图

- 候选 prompt：复用 3.2 的 CoT prompt；固定生成 5 个候选。
- 通用 selector：固定 commit 的 `self_consistency_gsm8k.py`，同文案也用于 MATH、
  HumanEval、MBPP。
- HotpotQA selector：`self_consistency_hotpotqa.py` 的 question+context 变体。
- DROP：固定 commit 无 self-consistency 文件，候选与 selector 均为
  `paper-faithful/inferred`。
- 可审计 selector 文本：`prompts/baselines/cot_sc/`。

主协议使用 5 次候选生成 + 1 次 selector。历史 HotpotQA 候选生成器内部先生成 thought、
再把 thought 回填后生成答案，因此历史调用图为 `5×2+1=11`；其他四个公开任务为 6 次。
为避免数据集间方法预算不同，Table 1 主复现统一为 6 次，11 次行为只归入
`archive-faithful`。

### 3.4 MedPrompt prompt 与调用图

- GSM8K 候选 prompt 和通用 voter：`medprompt_gsm8k.py`。
- MATH 候选 prompt 和通用 voter：`medprompt_math.py`。
- HumanEval/MBPP 候选 prompt 和 code voter：`medprompt_humaneval.py`、
  `medpromt_mbpp.py`（上游文件名原有拼写错误）。
- HotpotQA/DROP：固定 commit 无实现，候选沿用对应 CoT adapter，voter 为
  `paper-faithful/inferred`。
- 可审计新增文本：`prompts/baselines/medprompt/`；复用的候选文本仍位于
  `prompts/baselines/cot/`。

论文/其余公开入口的主结构是 3 个候选 + 5 次乱序投票。历史 MATH `__call__` 仅生成 2 个
候选，入口又传 `vote_count=2`，不能代表论文表中的 MedPrompt。本项目主协议统一 3+5，
并将 seed、每次 permutation、显示字母、原始候选索引和计票结果写入 artifacts。

### 3.5 MultiPersona prompt 与调用图

- GSM8K/MATH：历史固定三个数学角色；HotpotQA：历史固定三个知识问答角色；
  HumanEval/MBPP：历史固定三个编程角色。
- 每个角色先独立回答一轮，第二轮读取三个角色上一轮的 thinking，再由一个 synthesis
  调用读取第二轮的 thinking 与 answer；主协议每题固定 `3×2+1=7` 次请求。
- DROP 历史实现缺失，复用 HotpotQA 的问答角色与 GSM8K 的通用讨论结构，标记为
  `paper-faithful/inferred`。
- 历史 HumanEval/MBPP 第二轮调用把 context 列表位置绑定到 `function_name`，导致实际重复
  第一轮 prompt。主协议修复参数绑定，使 peer context 真正进入第二轮；异常路径只作为
  `archive-faithful` 记录。

历史 user prompt 保存在 `src/awo/baselines/multi_persona.py` 的冻结常量中；显式 JSON system
消息替代 ActionNode 的结构化注入，不增加调用。每次运行保存角色、两轮 thinking/answer、
prompt hash 和逐请求 provider 审计信息。

### 3.6 Self-Refine prompt 与调用图

- GSM8K/MATH/HumanEval/MBPP：生成、review、revise user prompt 来自历史固定提交；初始生成
  复用相应 CoT adapter。
- HotpotQA/DROP：历史实现缺失，复用任务 CoT 生成器、通用 review 和最小 QA revise
  adapter，标记为 `paper-faithful/inferred`。
- 每题先生成 1 次，然后最多执行 3 轮 review；review 返回 false 时同轮追加 1 次 revise。
  因此请求数为 2、4、6 或 7（第三轮接受为 6，第三轮拒绝并 revise 为 7）。
- review 只接受显式 JSON boolean（兼容字符串 `"true"`/`"false"`）；解析失败不以 Python
  字符串 truthiness 猜测。每轮保存修改前答案、feedback、判定和修改后答案。

可审计 review/revise 文本位于 `prompts/baselines/self_refine/`。历史实现将 Pydantic 导出的
整个 solution 字典格式化回后续 prompt；主协议显式使用其中的 solution 文本。

## 4. ADAS

- 仓库：https://github.com/ShengranHu/ADAS
- 固定提交：`2702bee8fefda42255efc5be9f60e3bd3db96ae4`
- 提交时间：2025-01-27T22:37:51-08:00
- 提交说明：`fix role description typo`
- 许可证：Apache-2.0

ADAS 上游公开了多个自包含 domain，但没有 AFlow Table 1 所用的六任务统一 adapter、初始 archive 和完整搜索配置。本项目将复用 Meta Agent Search，并将自建部分标记为 `protocol-compatible`。

固定 DROP 实现确认了以下公开算法细节：

- 初始 archive 为 Chain-of-Thought、Self-Consistency、Self-Refine、LLM Debate、
  Step-back Abstraction、Quality-Diversity 和 Dynamic Assignment of Roles；
- 每个生成代先 proposal，再在同一消息历史上连续做两次 reflection；
- 默认生成 30 代，meta-agent temperature 为 0.8；
- 每次候选评估后连同 fitness 加入完整 archive，下一代看到整个 archive；
- 官方候选是一个 Python `forward(self, taskInfo)`，通过宿主 `exec` 动态加载；
- 官方 DROP 默认 validation size 为 128，但 AFlow Table 1 六任务的 validation 取样和
  adapter 没有公开。

本项目保留七 seed、三次 meta 调用、完整 archive 和 30 代循环；六任务沿用已冻结的 AFlow
validation split，不另造 128 条子集。候选表示改为 `agent_dag_v1`，并由解释器逐节点调用
统一 OpenRouter 客户端。该安全适配及七 seed 的 DAG 翻译均明确属于
`protocol-compatible`。

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

## 7.1 AFlow 原版 Python 搜索与沙箱适配

HumanEval 原版搜索使用 `third_party/aflow_paper/` 中固定 commit 的 optimizer、prompt、
HumanEval 初始 graph 和 operator 源码。该目录中的上游文件保持字节不变；本地兼容层位于
`src/awo/aflow/official.py` 和 `src/awo/aflow/official_search.py`。

优化器继续接收官方拼接后的单条 user message，并按官方 `XmlFormatter` 的
`modification/graph/prompt` 标签解析。父节点选择、experience 去重、`sample=4`、
`alpha=0.2`、`lambda=0.3`、5 次验证及收敛判定均跟随固定源码。唯一传输替换是把官方
`AsyncOpenAI` 接到统一 OpenRouter 客户端。

模型生成的 `graph.py` 和 `prompt.py` 不在 AWO 主进程导入。它们连同固定 operator 源码在
无网络沙箱中执行；LLM 调用通过 transcript relay 交回宿主。最终 HumanEval 测试代码也由
同一沙箱评分。此适配改变隔离与 API 传输位置，不改 optimizer prompt 或候选 Python。

固定 commit 的 HumanEval `Test` 模板原文包含 `dataset="MBPP"`，并在反思分支读取
`response["reflection_and_solution"]`；后者直到上游 `b6e0bad` 才修复。原版路径不静默修复
这些已知上游行为，失败会写入对应 round 的日志。
