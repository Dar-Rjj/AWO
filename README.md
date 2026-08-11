# AWO：AFlow 主实验复现

本仓库用于复现论文 [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762) 的 Table 1 主实验及其 baselines。

本项目采用统一模型协议：AFlow optimizer、ADAS meta-agent、workflow operators、所有 baseline、reviewer、voter 和 selector 均通过 OpenRouter 调用 `deepseek/deepseek-chat`。客户端实现参考：

- `/home/rjj/RealEvo/utils/llm_client/openrouter.py`
- `/home/rjj/RealEvo/cfg/llm_client/openrouter.yaml`

> 当前状态：阶段 0（协议/工件冻结）、阶段 1（项目脚手架）、阶段 2
>（统一 OpenRouter 客户端）、阶段 3（数据与统一 evaluator）和阶段 4（六个手工
> baselines）已完成。真实 preflight 已确认 `deepseek/deepseek-chat` 可用；下一步为
> AFlow 主方法；operator/runtime 与历史 workflow 安全兼容层已完成，下一步为官方最佳
> workflow 的六任务重放。

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

当前实现使用 `docker/code-sandbox/Dockerfile`。供应链输入固定为 Python 3.9 基础镜像
digest `sha256:2d97…b1b` 和 NumPy 2.0.2 CPython 3.9 x86_64 wheel SHA256
`f26b…03dd`。NumPy 是六份官方最终代码 CSV 中唯一被历史通过预测实际依赖的第三方包；
使用 NLTK/SymPy 的历史预测原本均记 0，因此不额外扩张镜像。

```bash
docker build --pull \
  --tag awo-code-sandbox:py3.9-numpy2.0.2 \
  docker/code-sandbox

# 默认跳过真实 Docker 测试；显式启用后验证断网、只读、非 root、超时和输出上限
AWO_RUN_DOCKER_TESTS=1 pytest -q tests/security/test_docker_sandbox.py
python scripts/sandbox_check.py
```

运行器先核验镜像 labels，再把本地 tag 解析为不可变 Image ID。本机已验证的构建 ID 为
`sha256:3ac37a78e2e380ddb53b9c0ca9b672ffde319605113d97620fb92b38ff022861`；
该 ID 是本次运行 manifest 信息，不是跨平台镜像名称。每次容器均使用 `--network none`、
只读根文件系统、UID/GID 65534、丢弃全部 capabilities、`no-new-privileges`、独立 tmpfs，
并限制 CPU、内存、PID、文件大小、打开文件数、墙钟时间和合计输出大小。容器不挂载宿主目录。

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
│   ├── prepare_data.py
│   ├── replay_results.py
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

3B 已完成。12 个 validation/test 文件被确定性规范化为 4,515 条统一 schema 记录；
官方 HotpotQA、DROP、GSM8K、MATH 共 12 份最终 CSV 已逐行重评分，`archive` 模式
与全部原分数零差异。MATH 的 `source`、`archive`、`corrected` 三种模式及工件反推的
窄兼容规则见 `docs/ambiguities.md` A-015。

3C 已完成。代码提取、HumanEval/MBPP harness、逐样本 Docker runner 和归档 CSV replay
已实现；46 个单元测试和 5 个真实 Docker 安全测试通过。官方 HumanEval 首份最终 CSV
全量 131 条重评分只有 `HumanEval/99` 一条历史标签异常：当前为 124 pass / 7 fail、
pass@1 0.946565，归档为 0.938931；修正按配置联动的 CPU rlimit 后，`HumanEval/129`
在 6.79 秒正常通过。官方 MBPP 首份最终 CSV 全量 341 条重放为 285 pass / 56 fail、
pass@1 0.835777，与归档逐行零差异。详见 A-016。

### 阶段 4：手工 baselines

1. 依次实现 IO、CoT、CoT-SC、MedPrompt、MultiPersona 和 Self-Refine。
2. 移植历史 prompt，补齐 7 个缺失组合。
3. 对调用图、调用次数、停止条件和 parser 建立 golden tests。
4. 同时支持 `paper-faithful`、`archive-faithful` 和 `corrected-controlled` 模式。

验收：每个 baseline 可在每个数据集的 2 个样本上完整运行，并满足预期调用预算。

进度（2026-08-09）：IO 已完成。固定 commit 中存在的 HotpotQA、GSM8K、MATH、
HumanEval、MBPP user prompt 已按原文移植；缺失的 DROP prompt 标记为
`paper-faithful/inferred`。原 MetaGPT `ActionNode` 的结构化填充由显式单字段 JSON system
schema 兼容替代，不增加 LLM 调用。六任务均强制 `call_count=1`，prompt 文本与运行时模板
有一致性测试。

真实单样本 smoke 均由 OpenRouter 路由至 `deepseek/deepseek-chat`（provider `Novita`）：

- IO–GSM8K：251 tokens，费用 `$0.0001634`，答案 36，得分 1.0；
- IO–HumanEval（最终上游 user prompt）：206 tokens，费用 `$0.0001364`，生成代码在固定
  Docker Image ID 中通过官方测试，得分 1.0。

两次最终 smoke 均为一次请求、一次成功，审计日志未包含凭据字段。单样本成功仅验证链路，
不作为数据集效果估计；阶段 7 仍会按预注册规模执行 smoke/pilot。

CoT 已完成。HotpotQA、GSM8K、MATH、HumanEval、MBPP 的历史 user prompt 已移植；
`cot_drop.py` 的中文 prompt 与实际 evaluator 参数不兼容且入口只跑 1 条 validation，主协议
采用英文 task adapter 并标记 `paper-faithful/inferred`。六任务均保持每题 1 次生成调用，
不增加独立 rationale 或答案提取请求。真实 CoT–GSM8K 单样本 smoke 使用 273 tokens、费用
`$0.00013125`，答案 36、得分 1.0；requested/actual model 均为
`deepseek/deepseek-chat`，但 provider 为 `DeepInfra`，不同于此前 IO 的 `Novita`。

CoT-SC 已完成。主协议固定为 5 个独立 CoT 候选 + 1 个 LLM selector，每题严格 6 次
请求；候选按 A–E 编号，selector 非法输出 fail-closed，不默认选择 A。历史 HotpotQA 文件
的每个候选实际使用两次生成、总计 11 次请求，该行为不进入跨任务公平主表，仅作为
`archive-faithful` 差异记录。DROP 候选和 selector 均标为 inferred。

真实 CoT-SC–GSM8K 单样本 5+1 smoke 的 6 次请求全部一次成功，共 2,254 tokens、费用
`$0.00115436`；5 个候选均得到 36，selector 选择 A，得分 1.0。同一题内 provider 分布为
`DeepInfra: 4, Novita: 2`，因此 run manifest 必须按请求记录 provider，而不能只在 run
级别记录一个值。

MedPrompt 已完成。主协议固定为 3 个独立候选 + 5 次独立乱序投票，每题严格 8 次请求；
每次排列由 `seed + sample_id` 确定并记录，票的 A/B/C 显式映射回原始候选索引，五票平局
固定选择最低原始索引。任一票无法解析则样本 fail-closed，不静默减少投票分母。历史 MATH
入口实际为 2 候选 + 2 投票，HotpotQA/DROP 缺失；这些不一致不进入 3+5 主表。

真实 MedPrompt–GSM8K 3+5 smoke 的 8 次请求全部一次成功，共 2,347 tokens、费用
`$0.0008207856`；五票映射为 `[2,2,1,1,1]`，候选 1 以 3 票胜出，答案 36、得分 1.0；
provider 分布为 `DeepInfra: 4, StreamLake: 4`。

MultiPersona 已完成受控实现。六任务统一为 3 个固定角色 × 2 轮讨论 + 1 次综合，每题严格
7 次请求；第二轮显式读取三名角色第一轮 thinking。HumanEval/MBPP 历史 context 位置参数
错绑已在主协议修复，DROP adapter 明确标记为 inferred；逐轮内容、角色和 prompt hash 全部
进入结果 artifacts。

真实 MultiPersona–GSM8K smoke 的 7 次请求全部一次成功，共 4,329 tokens、费用
`$0.0023651174`，综合答案 36、得分 1.0；provider 分布为
`DeepInfra: 2, Novita: 4, StreamLake: 1`，审计记录未包含凭据。

Self-Refine 已完成受控实现。每题先生成一次，然后最多运行 3 轮 review→revise，请求数随
提前接受而变化，范围为 2–7。review_result 采用显式布尔解析，字符串 `"false"` 不会因
Python truthiness 被误判为接受；HotpotQA/DROP adapter 标记为 inferred，每轮答案、反馈、
修订和停止原因均进入 artifacts。

真实 Self-Refine–GSM8K smoke 在首轮 review 接受，共 2 次请求、644 tokens、费用
`$0.0002797655`，答案 36、得分 1.0；provider 分布为 `DeepInfra: 1, StreamLake: 1`，
两次请求均一次成功且审计记录未包含凭据。

### 阶段 5：AFlow

1. 移植并测试 operator。
2. 建立旧工件兼容层。
3. 重放官方最佳 workflow 和轨迹。
4. 实现 20 轮搜索、候选选择、经验树和早停。
5. 运行 alpha/lambda 敏感性配置。

当前已完成第一子步骤：六任务所需的 6 类 operator 及统一异步 runtime 已移植。所有 LLM
调用进入 OpenRouter 审计与成本账本；`Programmer`/`Test` 强制使用锁定的 Docker 沙箱，
不再执行上游宿主 `exec`；代码 entry point 丢失问题已修复并记录为 A-022。

真实 AFlow blank workflow–GSM8K smoke 使用 `Custom` 单次调用，共 312 tokens、费用
`$0.00021213`，答案 36、得分 1.0；provider 为 `DeepInfra`，请求一次成功且审计记录未
包含凭据。

第二子步骤已完成：官方 `graphs_test` 六个最佳 round 及其 graph/prompt SHA256 已冻结；兼容
层不动态执行混用旧导入路径的归档 graph.py，而在哈希验证后以原生白名单 adapter 重建同一
operator 拓扑。prompt.py 仅作 AST 字符串字面量读取，代码任务公开测试仍强制进入 Docker。

真实 GSM8K 官方最佳 round 10 重放严格执行 `5×Custom + ScEnsemble + Programmer`，共
7 次请求、4,154 tokens、费用 `$0.0020009917`；Programmer 生成代码在锁定 Docker 中一次
执行成功，输出 36、得分 1.0。provider 分布为
`DeepInfra: 3, Novita: 1, StreamLake: 3`；请求全部一次成功且审计记录未包含凭据。

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
python scripts/prepare_data.py

# 如需同时下载论文结果和初始 workflow（默认只下载 datasets）
python scripts/download_data.py --artifact all

# 重放官方 QA/数学结果（需先下载 results）
python scripts/replay_results.py --results-dir data/results

# 构建并验证代码评测镜像
docker build --pull --tag awo-code-sandbox:py3.9-numpy2.0.2 docker/code-sandbox
AWO_RUN_DOCKER_TESTS=1 pytest -q tests/security/test_docker_sandbox.py

# 重放单份官方代码结果
python scripts/replay_code_results.py HumanEval RESULT.csv data/raw/datasets/humaneval_test.jsonl
python scripts/replay_code_results.py MBPP RESULT.csv data/raw/datasets/mbpp_test.jsonl

# 单条 IO 真实链路 smoke
python scripts/run_io_smoke.py gsm8k data/raw/datasets/gsm8k_validate.jsonl \
  --record-file experiments/io-gsm8k-smoke.jsonl

# 单条 CoT 真实链路 smoke
python scripts/run_cot_smoke.py gsm8k data/raw/datasets/gsm8k_validate.jsonl \
  --record-file experiments/cot-gsm8k-smoke.jsonl

# 单条 CoT-SC 5+1 真实链路 smoke
python scripts/run_cot_sc_smoke.py gsm8k data/raw/datasets/gsm8k_validate.jsonl \
  --record-file experiments/cot-sc-gsm8k-smoke.jsonl

# 单条 MedPrompt 3+5 真实链路 smoke
python scripts/run_medprompt_smoke.py gsm8k data/raw/datasets/gsm8k_validate.jsonl \
  --seed 0 --record-file experiments/medprompt-gsm8k-smoke.jsonl

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
