# AWO：AFlow 主实验复现

本仓库用于复现论文 [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762) 的 Table 1 主实验及其 baselines。

本项目采用统一模型协议：AFlow optimizer、ADAS meta-agent、workflow operators、所有 baseline、reviewer、voter 和 selector 均通过 OpenRouter 调用 `deepseek/deepseek-chat`。客户端实现参考：

- `/home/rjj/RealEvo/utils/llm_client/openrouter.py`
- `/home/rjj/RealEvo/cfg/llm_client/openrouter.yaml`

> 当前状态：阶段 0（协议/工件冻结）、阶段 1（项目脚手架）和阶段 2
>（统一 OpenRouter 客户端）已完成。真实 preflight 已确认
> `deepseek/deepseek-chat` 可用；阶段 3A（工件下载、哈希校验和安全解压）已完成，
> evaluator 与代码安全沙箱正在实现。

## 1. 复现范围

第一阶段以论文 Table 1 为验收范围：

- 6 个数据集：HotpotQA、DROP、HumanEval、MBPP、GSM8K、MATH Level 5。
- 6 个手工 baseline：IO、CoT、CoT-SC、MedPrompt、MultiPersona、Self-Refine。
- 1 个自动工作流 baseline：ADAS。
- 主方法：AFlow。
- 每个方法在每个数据集测试集上独立运行 3 次。
- 输出论文分数、复现均值、标准差、95% 置信区间、差值、调用量、token、延迟、错误数和费用。

暂不纳入第一阶段验收的内容：Table 2 跨模型实验、消融实验、搜索曲线和论文其他图表。这些内容将在 Table 1 链路稳定后扩展。

### 1.1 论文目标值

下表仅作为比较基准。LLM 服务、模型路由和公开工件存在版本差异，复现成功不以强行追平每一个数值为条件。

| 方法 | HotpotQA | DROP | HumanEval | MBPP | GSM8K | MATH | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IO | 68.1 | 68.3 | 87.0 | 71.8 | 92.7 | 48.6 | 72.8 |
| CoT | 67.9 | 78.5 | 88.6 | 71.8 | 92.4 | 48.8 | 74.7 |
| CoT-SC | 68.9 | 78.8 | 91.6 | 73.6 | 92.7 | 50.4 | 76.0 |
| MedPrompt | 68.3 | 78.0 | 91.6 | 73.6 | 90.0 | 50.0 | 75.3 |
| MultiPersona | 69.2 | 74.4 | 89.3 | 73.6 | 92.8 | 50.8 | 75.1 |
| Self-Refine | 60.8 | 70.2 | 87.8 | 69.8 | 89.6 | 46.1 | 70.7 |
| ADAS | 64.5 | 76.6 | 82.4 | 53.4 | 90.8 | 35.4 | 67.2 |
| AFlow | 73.5 | 80.6 | 94.7 | 83.4 | 93.5 | 56.2 | 80.3 |

## 2. 与论文原始设置的差异

论文 Table 1 使用 Claude 3.5 Sonnet 作为 AFlow/ADAS optimizer，使用 `gpt-4o-mini-2024-07-18` 作为 executor。本项目按复现要求统一替换为 OpenRouter 的 `deepseek/deepseek-chat`。

因此本项目属于：

- 数据、算法、prompt 和评测协议复现；
- 同模型条件下的公平受控比较；
- 不是论文原始模型组合的逐输出 exact reproduction。

最终报告必须同时列出“论文配置”和“本项目配置”，不能将 DeepSeek Chat 的结果标记为原论文模型复现结果。

OpenRouter 的 `deepseek/deepseek-chat` 是模型别名，后端版本或路由可能变化。每个 run 必须记录：

- 请求模型名及 API 返回的实际模型名；
- OpenRouter 路由/provider 信息（如果响应提供）；
- 请求时间、temperature、max tokens 和其他采样参数；
- prompt、配置和代码的 SHA256/Git commit。

## 3. 统一 LLM 调用协议

### 3.1 默认配置

计划配置文件：`configs/models/openrouter_deepseek_chat.yaml`。

```yaml
provider: openrouter
model: deepseek/deepseek-chat
base_url: https://openrouter.ai/api/v1
temperature: 1.0
n: 1
timeout_seconds: 120
max_retries: 8
```

选择 `temperature: 1.0` 的原因：RealEvo 的参考配置使用该值，论文涉及 DeepSeek 的实验也采用温度 1。所有 headline 方法使用同一个温度，避免方法间模型参数不一致。温度 0 只作为额外敏感性实验，不混入主表。

### 3.2 凭据

```bash
export OPENROUTER_API_KEY="YOUR_KEY"
```

API key 不得写入配置、日志、响应缓存或 Git。仓库只提交 `.env.example`。

### 3.3 客户端实现要求

客户端复用 RealEvo 的设计：

- 通过 OpenAI-compatible Chat Completions API 调用；
- 默认 `base_url=https://openrouter.ai/api/v1`；
- 从 `OPENROUTER_API_KEY` 读取凭据；
- 单次请求只使用 `n=1`；
- CoT-SC、MedPrompt 等多候选方法通过多次独立请求实现。

在此基础上增加实验所需能力：

- 有上限的指数退避重试，而不是无限重试；
- 429、5xx、连接失败、超时和无效响应分类统计；
- 请求/响应 ID、token、延迟和费用记录；
- JSON/结构化输出解析失败时保留原始响应；
- 可配置并发、rate limit、断点续跑和逐样本落盘；
- 每次重试不改变 prompt 和采样参数；
- 最终失败样本按 0 分计入原始分母，不静默丢弃。

### 3.4 模型角色

以下角色全部使用同一个 `OpenRouterClient` 和同一模型：

| 组件 | 角色 |
| --- | --- |
| AFlow optimizer | 生成、修改和选择 workflow 代码 |
| AFlow executor | 执行 workflow 内的所有 LLM operator |
| ADAS meta-agent | 生成新的 agent/workflow 程序 |
| ADAS executor | 执行候选 agent |
| Baselines | IO、CoT 和所有复杂 baseline 的生成调用 |
| Voter/selector | CoT-SC、MedPrompt、ensemble 的选择调用 |
| Reviewer/reviser | Self-Refine 的反馈和修改调用 |

## 4. 数据协议

优先使用 AFlow 官方发布的数据文件，不从上游数据集重新随机生成 split。

| 数据集 | Validation | Test | 指标 | 备注 |
| --- | ---: | ---: | --- | --- |
| HotpotQA | 200 | 800 | token F1 | 论文从数据集中抽取 1,000 条 |
| DROP | 200 | 800 | token F1 | 论文从数据集中抽取 1,000 条 |
| HumanEval | 33 | 131 | pass@1 | 总计 164 条 |
| MBPP | 86 | 341 | pass@1 | 官方工件实际总计 427 条 |
| GSM8K | 264 | 1,055 | solve rate | 总计 1,319 条 |
| MATH | 119 | 486 | solve rate | 官方工件 605 条；论文正文写 617 条 |

数据准备步骤：

1. 下载官方数据档案至 `data/raw/`。
2. 记录下载 URL、文件大小和 SHA256。
3. 验证上述样本数量和唯一 ID。
4. 检查 validation/test 是否有重复或泄漏。
5. 将 schema 规范化到 `data/processed/`，但不改写原始文件。
6. 将所有差异写入 `docs/ambiguities.md`。
7. 搜索和选择 workflow 时只能读取 validation；test 只在 workflow 冻结后使用。

## 5. 统一评测器

所有方法必须使用同一套 evaluator，禁止每个 baseline 自带略有不同的评分实现。

### 5.1 HotpotQA 和 DROP

- 小写化、去标点、去冠词并规范化空白；
- 计算预测与 gold answer 的 token precision、recall 和 F1；
- 多个 gold answer 取最大匹配分数。

### 5.2 GSM8K

- 从约定答案字段或模型输出末尾提取数值；
- 处理逗号、负数、小数、百分号和科学计数法；
- 使用官方实现兼容的数值容差；
- 无法解析时记为错误，不调用另一个 LLM 判分。

### 5.3 MATH

- 优先提取最后一个 `\boxed{}`；
- 其次使用末句答案规则；
- 使用符号化简检查表达式等价；
- parser 异常保留原始答案并记为失败。

### 5.4 HumanEval 和 MBPP

- 只评价生成函数，不允许模型访问测试答案；
- 每个样本在独立安全沙箱运行；
- 使用数据自带测试，指标为 pass@1；
- 编译错误、运行错误、超时和沙箱违规均记为失败。

### 5.5 Golden tests

- 为每个 evaluator 建立正常、边界和错误样例；
- 使用官方归档 CSV 重算论文公开结果；
- 只有逐样本评分与归档兼容后，才能运行付费实验。

## 6. Baselines

历史实现主要参考 MetaGPT commit [`22e8f9d7fcfabd4c1fc6640235079c1c70bc5899`](https://github.com/FoundationAgents/MetaGPT/tree/22e8f9d7fcfabd4c1fc6640235079c1c70bc5899/examples/ags/experiments/baselines)。当前独立 AFlow 仓库的 `run_baseline.py` 不是完整 Table 1 实现。

| 方法 | 调用结构 | 典型单题请求数 |
| --- | --- | ---: |
| IO | 一次任务特定直接生成 | 1 |
| CoT | 一次 step-by-step 生成 | 1 |
| CoT-SC | 5 个 CoT 候选，再由 LLM 选择 | 6 |
| MedPrompt | 3 个候选、5 次乱序投票 | 8 |
| MultiPersona | 3 persona × 2 轮，再汇总 | 7 |
| Self-Refine | 初始答案，最多 3 轮 review → revise | 2–7 |

### 6.1 实现细节

- **IO**：一次任务特定生成，输出满足各 evaluator 的格式约定。
- **CoT**：一次 step-by-step 生成，推理文本和最终答案分别保存。
- **CoT-SC**：生成 5 个独立候选，再请求 1 次 DeepSeek Chat selector；记录候选归一化答案和多样性。
- **MedPrompt**：生成 3 个候选，打乱候选顺序后投票 5 次；固定 tie-break 和 shuffle seed。
- **MultiPersona**：3 个固定任务角色各讨论 2 轮，再进行一次 synthesis。
- **Self-Refine**：初始生成后最多执行 3 轮 review → revise；保存每轮反馈、修改和停止原因。

### 6.2 缺失适配

历史公开代码缺少以下 7 个组合：

- IO–DROP；
- CoT-SC–DROP；
- MedPrompt–HotpotQA、DROP；
- MultiPersona–DROP；
- Self-Refine–HotpotQA、DROP。

补齐时必须保持方法调用图和调用预算不变，只替换任务描述、输入字段和输出 schema。所有推断 prompt 写入 `docs/provenance.md` 并标记为 `paper-faithful/inferred`。

同时保留三种运行模式：

- `paper-faithful`：论文协议优先，作为主报告；
- `archive-faithful`：保留历史 prompt 和已知异常；
- `corrected-controlled`：修复明确 bug，用于受控比较。

## 7. AFlow 复现

### 7.1 Operator 集合

| 数据集 | 可用 operators |
| --- | --- |
| HotpotQA、DROP | Custom、AnswerGenerate、ScEnsemble |
| GSM8K、MATH | Custom、ScEnsemble、Programmer |
| HumanEval、MBPP | Custom、CustomCodeGenerate、ScEnsemble、Test |

### 7.2 搜索协议

- 初始 workflow：论文 blank/template workflow；
- optimizer 和 workflow executor 均使用 `deepseek/deepseek-chat`；
- 最大搜索轮数：20；
- 每个新 workflow 在 validation 上重复评估 5 次；
- 候选池 top-k：3；
- 连续 5 轮未改善时允许提前停止；
- 保存每轮 workflow 代码、prompt、父节点、修改说明、错误反馈、validation 分数和选择概率；
- 选择最终 workflow 后冻结代码，再执行 3 次独立 test run。

### 7.3 论文参数冲突

AFlow 混合选择参数存在三种公开值：

- 论文正文：`alpha=0.4, lambda=0.2`；
- 论文附录：`alpha=0.2, lambda=0.4`；
- 当前官方代码：`alpha=0.2, lambda=0.3`。

主实验使用论文正文值，另外两组作为敏感性实验。三组结果分别保存，不从中挑选最好的一组作为主结果。

### 7.4 工件重放

完整搜索前先：

1. 下载官方 AFlow 搜索轨迹和最佳 workflow。
2. 通过兼容层修复旧 `examples.aflow`/`metagpt.ext.aflow` 导入路径。
3. 使用统一 evaluator 重算归档分数。
4. 使用 DeepSeek Chat 重新执行最佳 workflow。
5. 将“归档答案重评分”和“新模型重新推理”分开报告。

## 8. ADAS 复现

参考 [ADAS 官方仓库](https://github.com/ShengranHu/ADAS) 的 Meta Agent Search。

- meta-agent 和候选 agent executor 均使用 `deepseek/deepseek-chat`；
- 搜索 30 轮；
- 使用与 AFlow 相同的 validation/test split 和 evaluator；
- 为六个数据集实现统一 adapter；
- 搜索过程只能访问 validation；
- 保存每轮生成代码、父代、prompt、运行错误和分数；
- 最优 agent 冻结后执行 3 次 test run。

AFlow 论文所用的 ADAS 六任务适配器没有公开，因此该结果标记为 `protocol-compatible`，不能描述为原作者代码的 exact reproduction。

## 9. 生成代码安全

HumanEval、MBPP、AFlow Programmer 和 ADAS 会执行模型生成代码。禁止直接在主进程或宿主环境中调用不受限的 `exec`。

每个样本必须在独立沙箱中运行：

- 非 root 用户；
- 禁用网络；
- 只读根文件系统；
- 独立临时工作目录；
- 限制 CPU、内存、PID、文件大小和执行时间；
- 超时后终止整个进程组；
- 不挂载 API key、SSH key、Git 凭据和可写项目目录；
- stdout/stderr 截断后作为实验工件保存。

安全测试通过前，不运行任何模型生成代码。

## 10. 计划目录

```text
AWO/
├── configs/
│   ├── models/openrouter_deepseek_chat.yaml
│   ├── paper/table1.yaml
│   ├── smoke.yaml
│   └── pilot.yaml
├── src/awo/
│   ├── llm/
│   ├── benchmarks/
│   ├── baselines/
│   ├── workflows/aflow/
│   ├── workflows/adas/
│   ├── sandbox/
│   └── tracking/
├── prompts/
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── third_party/
├── patches/
├── scripts/
│   ├── preflight.py
│   ├── download_data.py
│   ├── run_table1.py
│   ├── aggregate.py
│   └── verify_artifacts.py
├── experiments/
├── reports/
├── tests/
│   ├── unit/
│   ├── golden/
│   ├── integration/
│   └── security/
└── docs/
    ├── protocol.md
    ├── provenance.md
    └── ambiguities.md
```

## 11. 分阶段实施步骤

### 阶段 0：冻结复现规格和来源

1. 固定论文 v4。
2. 固定论文同期 AFlow commit、当前参考 commit 和 MetaGPT baseline commit。
3. 下载官方数据及结果档案并记录 SHA256。
4. 建立 `protocol.md`、`provenance.md` 和 `ambiguities.md`。
5. 记录 MATH 数量、AFlow 参数、历史 prompt 和 ADAS 适配等公开冲突。

验收：每个数据、prompt、算法设置都能追溯到论文、公开代码或明确标记的推断。

### 阶段 1：仓库与环境

1. 创建独立 Python 3.9 环境。
2. 建立 `pyproject.toml` 和锁文件。
3. 实现配置加载、日志、run manifest 和 CLI。
4. 添加 `.gitignore`、`.env.example`、许可证说明和基础 CI。

验收：新环境可安装，单元测试框架可运行，仓库中不存在密钥。

### 阶段 2：OpenRouter 客户端

1. 参考 RealEvo 实现 `OpenRouterClient`。
2. 增加 bounded retry、timeout、并发限制和使用量记录。
3. 实现 `preflight`：发送最小请求并验证模型、响应结构和 token usage。
4. 将 optimizer、executor 和 baselines 注入同一个 client factory。

验收：所有角色的请求日志均显示 `deepseek/deepseek-chat`，且失败不会无限重试。

完成记录（2026-08-09）：离线检查、15 项单元测试、lint 和 Python 3.9 编译通过；
一次最小真实请求由 OpenRouter 路由至 `deepseek/deepseek-chat`（provider `Novita`），
共 27 tokens、一次成功，审计日志中未出现 API key。provider 属于运行时信息，后续每次
实验仍以请求日志中的实际返回值为准。

### 阶段 3：数据与 evaluator

1. 实现幂等下载和 hash 校验。
2. 规范化六个数据集 schema。
3. 实现六类 evaluator。
4. 建立单元测试和归档 CSV golden tests。
5. 实现 HumanEval/MBPP 安全沙箱。

验收：官方归档结果可被重新评分，沙箱安全测试全部通过。

进度（2026-08-09）：3A 已完成。真实数据归档通过 SHA256/大小校验，15 个文件的
哈希与记录数全部通过，安全解压的幂等、路径穿越和链接拒绝测试通过。

### 阶段 4：手工 baselines

1. 依次实现 IO、CoT、CoT-SC、MedPrompt、MultiPersona 和 Self-Refine。
2. 移植历史 prompt，补齐 7 个缺失组合。
3. 对调用图、调用次数、停止条件和 parser 建立 golden tests。
4. 同时支持 `paper-faithful`、`archive-faithful` 和 `corrected-controlled` 模式。

验收：每个 baseline 可在每个数据集的 2 个样本上完整运行，并满足预期调用预算。

### 阶段 5：AFlow

1. 移植并测试 operator。
2. 建立旧工件兼容层。
3. 重放官方最佳 workflow 和轨迹。
4. 实现 20 轮搜索、候选选择、经验树和早停。
5. 运行 alpha/lambda 敏感性配置。

验收：可从初始模板生成、评估、选择并持久化新 workflow；测试数据不进入搜索。

### 阶段 6：ADAS

1. 移植 Meta Agent Search。
2. 为六个数据集实现 adapter 和 task prompt。
3. 接入统一 evaluator 和安全沙箱。
4. 实现 30 轮搜索及最优 agent 冻结。

验收：六个任务都能完成小规模搜索，所有自建适配均有 provenance 标签。

### 阶段 7：Smoke 和 Pilot

Smoke：

- 每数据集 2–5 条样本；
- 运行 IO、一个复杂 baseline、AFlow 最佳归档 workflow 和 ADAS 单轮；
- 目标是发现 schema、parser、prompt 和沙箱问题。

Pilot：

- 每数据集/方法约 20 条样本；
- 检查模型输出多样性、重试率、限流、token、费用和吞吐；
- 依据真实 token 分布估算全量成本；
- 在用户确认预算后才进入全量运行。

### 阶段 8：完整 Table 1

1. 锁定代码 commit、配置和数据 manifest。
2. AFlow/ADAS 只在 validation 上搜索。
3. 冻结最佳 workflow/agent。
4. 每个方法在完整 test split 上执行 3 次独立 run。
5. paper run 禁止跨 run 复用模型响应缓存。
6. 允许断点续跑，但不能更换模型、prompt 或采样参数。

### 阶段 9：聚合与报告

每个 dataset/method 输出：

- 论文分数；
- 3 次复现分数；
- mean、standard deviation、95% CI；
- 与论文的绝对差值；
- 成功、失败、解析失败和超时数量；
- LLM 请求数、输入/输出 token、延迟和实际费用；
- AFlow/ADAS 搜索成本与最终 workflow 执行成本；
- 请求模型与实际路由模型；
- 复现等级和已知偏差。

结果差异超过 2 个百分点时执行样本级诊断，但不修改结果或针对测试集调 prompt。

## 12. 目标命令

以下命令将在对应实现完成后提供：

```bash
cd /home/rjj/AWO

# 安装
conda create -n awo python=3.9 -y
conda activate awo
python -m pip install pip==25.2
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .

# 模型和环境预检
python scripts/preflight.py --config configs/models/openrouter_deepseek_chat.yaml

# 下载并校验数据/工件
python scripts/download_data.py
python scripts/verify_artifacts.py

# 如需同时下载论文结果和初始 workflow（默认只下载 datasets）
python scripts/download_data.py --artifact all

# 单元、golden 和安全测试
pytest -q

# Smoke
python scripts/run_table1.py --config configs/smoke.yaml

# Pilot
python scripts/run_table1.py --config configs/pilot.yaml

# 完整实验；在 pilot 预算确认后运行
python scripts/run_table1.py --config configs/paper/table1.yaml

# 聚合报告
python scripts/aggregate.py --runs experiments/runs --output reports/table1
```

## 13. 运行记录与可复现性

每次实验生成不可变 run 目录：

```text
experiments/runs/<run_id>/
├── manifest.json
├── config.resolved.yaml
├── predictions.jsonl
├── requests.jsonl
├── metrics.json
├── errors.jsonl
└── artifacts/
```

`manifest.json` 至少包含：

- Git commit 和 dirty 状态；
- 数据文件 SHA256；
- prompt SHA256；
- 请求模型、实际模型和 provider；
- 完整采样参数；
- baseline/workflow 版本；
- validation/test seed；
- 开始、结束时间；
- Python 和依赖版本；
- 主机及沙箱信息。

## 14. 预算与运行门槛

六个手工 baseline 在 3,613 个测试样本上独立运行 3 次，预计可能超过 30 万次 LLM 请求；这还不包括：

- AFlow 约 94,710 次 validation workflow 执行，每个 workflow 内可能包含多次 LLM 请求；
- ADAS 30 轮搜索；
- 失败重试；
- 最终 AFlow/ADAS 三次完整测试。

因此执行顺序必须是：单元测试 → smoke → pilot → 费用报告 → 预算确认 → 全量实验。不得跳过 pilot 直接启动全部 API 请求。

## 15. 完成标准

- 六个数据集的文件数量、split 和 SHA256 可验证；
- 统一 evaluator 通过归档结果和边界样例测试；
- 所有模型调用均走 OpenRouter `deepseek/deepseek-chat`；
- 所有方法使用相同模型配置和错误计分规则；
- validation/test 无泄漏；
- 生成代码只能在安全沙箱运行；
- 8 个方法 × 6 个数据集均有 3 次独立测试结果；
- 结果包含统计量、token、费用、失败样本和复现等级；
- 公开缺失或推断实现均明确记录，不将 compatible 结果标记为 exact；
- README 中的安装、smoke、完整运行和聚合命令可从干净环境执行。
