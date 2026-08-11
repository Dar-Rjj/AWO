# 已知歧义与处理决策

本文件是协议的一部分。状态取值：

- `open`：尚无足够证据；
- `resolved-for-primary`：主实验已有预注册决策，但原始事实仍不唯一；
- `sensitivity`：主实验外另跑敏感性配置；
- `artifact-only`：只能重放工件，不能重建原始过程。

## A-001：模型替换

- 状态：`resolved-for-primary`
- 论文：Claude 3.5 Sonnet optimizer + GPT-4o-mini executor。
- 本项目：所有角色统一为 OpenRouter `deepseek/deepseek-chat`，temperature 1.0。
- 决策：报告为统一模型协议复现，禁止标记为论文模型 exact reproduction。

## A-002：OpenRouter 模型别名漂移

- 状态：`open`
- 问题：`deepseek/deepseek-chat` 可能随时间解析到不同后端版本/provider。
- 决策：每次请求记录请求模型、返回模型、路由 provider 和时间；preflight 结果写入 run manifest。不同实际模型不得聚合为同一次实验。
- 2026-08-09 证据：preflight 和 IO smoke 路由到 `Novita`，随后 CoT smoke 路由到
  `DeepInfra`；三者 requested/actual model 字段均为 `deepseek/deepseek-chat`。最终报告需列出
  provider 分布，不能把 provider 当作固定常量。
- CoT-SC 单题证据：6 次请求内部已同时出现 `DeepInfra` 4 次和 `Novita` 2 次；run-level
  单个 provider 字段不足以表达实际路由，必须从 request log 聚合分布。

## A-003：MATH 样本数量

- 状态：`resolved-for-primary`
- 论文：四类 Level 5 共 617 条。
- 官方数据工件：119 validation + 486 test = 605 条。
- 决策：主实验使用官方工件 605 条，不补造 12 条；报告显式列出差异。

## A-004：高方差 validation 子集

- 状态：`open`
- 论文：blank workflow 在 validation 执行 5 次，再选择高方差问题。
- 缺失：方差阈值、最终 ID 列表和完整生成规则。
- 决策：优先从结果/初始轮次工件恢复 ID；恢复失败时主实验使用完整 validation，并标记 `paper-faithful/inferred`。

## A-005：AFlow alpha/lambda

- 状态：`sensitivity`
- 正文：`alpha=0.4, lambda=0.2`。
- 附录 Algorithm 1：`alpha=0.2, lambda=0.4`。
- 当前代码：`alpha=0.2, lambda=0.3`。
- 决策：正文值为主实验；另外两组单独运行敏感性实验。

## A-006：validation 重复次数

- 状态：`resolved-for-primary`
- 论文与官方 README：每个新 workflow 5 次。
- 当前 `run.py` 默认：`validation_rounds=1`。
- 决策：主实验固定为 5，不采用当前 CLI 默认值。

## A-007：sample=4 与 top-k=3

- 状态：`open`
- 论文附录：top-k=3。
- 当前 `run.py`：`sample=4`。
- 可能解释：3 个历史候选加 blank workflow，但公开实现未给出唯一说明。
- 决策：搜索选择只允许 3 个 top workflow，并单独保留 blank/template 的探索概率。

## A-008：AFlow 轮数边界

- 状态：`open`
- 论文：20 iteration rounds。
- 部分结果目录出现 21 以上轮号，可能包含初始轮或续跑。
- 决策：run manifest 分开记录 initial evaluation 和 20 个 generated rounds，不把初始模板计为生成轮。

## A-009：缺失 baseline 组合

- 状态：`resolved-for-primary`
- 缺失 7 组：IO–DROP、CoT-SC–DROP、MedPrompt–HotpotQA/DROP、MultiPersona–DROP、Self-Refine–HotpotQA/DROP。
- 决策：保持同方法调用图和预算，仅适配任务字段和输出 schema；prompt 标记为 inferred。

## A-010：历史 baseline 异常

- 状态：`sensitivity`
- `cot_drop.py` 为明显临时配置。
- 历史 MedPrompt MATH 入口与论文 3 answers + 5 votes 不一致。
- HumanEval/MBPP MultiPersona 存在 peer context 位置参数绑定问题。
- 决策：主表使用 paper-faithful；另保留 archive-faithful 和 corrected-controlled。

## A-011：CoT-SC 名称

- 状态：`resolved-for-primary`
- 表格“5-shot”实际指生成 5 个 answers，不是 5 个 few-shot exemplars。
- 历史实现使用 LLM selector，而非规范化答案的确定性多数票。
- 决策：5 次候选 + 1 次 DeepSeek selector，并保存候选多样性。

## A-012：ADAS 六任务适配

- 状态：`artifact-only`
- AFlow 使用的 ADAS 六任务 adapter、初始 archive 和完整 prompt 未公开。
- 决策：基于 ADAS 固定 commit 实现兼容 adapter，结果标记 `protocol-compatible`。

## A-013：官方结果 CSV

- 状态：`resolved-for-primary`
- 部分 CSV 在真正表头前包含额外一行分数/时间标记。
- 决策：解析器显式识别 schema 和表头，不依赖固定首行；原始文件只读保留。

## A-014：代码执行安全

- 状态：`resolved-for-primary`
- 上游 AFlow/ADAS 会执行模型生成代码。
- 决策：禁止宿主进程直接 `exec`；所有代码任务和生成程序在无网络、非 root、受资源限制的独立沙箱运行。

## A-015：MATH 归档分数与公开 evaluator

- 状态：`resolved-for-primary`
- 现象：公开源码及其已声明依赖可重放绝大多数 MATH 行，但三份最终 CSV 仍有
  4/1/1 条记录不一致；差异仅涉及转义百分号和元组/矩阵内部空白。
- 证据：这些记录在官方 CSV 中均记 1；按公开源码的 `parse_expr` 回退会记 0；
  加装未声明的 ANTLR 则额外改变 90 条记录，不能解释归档环境。
- 决策：`source` 模式原样保留公开 evaluator；主实验 `archive` 模式只增加百分号显示值、
  裸元组空白和 `pmatrix` 空白兼容，以逐行重现官方 CSV；`corrected` 模式单独报告。

## A-016：代码归档环境与单条 HumanEval 标签

- 状态：`resolved-for-primary`
- 环境差异：MBPP 最终 CSV 中 `mbpp-721` 的三份预测均使用 NumPy 且归档记 1；纯
  `python:3.9-slim` 无 NumPy，会错误重放为 0。NLTK/SymPy 预测在归档中本来均记 0。
- 超时差异：固定 5 秒 CPU rlimit 会错误终止归档通过的 `HumanEval/129`；该预测在
  10 秒协议下耗时 6.79 秒通过。
- 历史标签：`HumanEval/99` 的归档预测对冻结测试全部通过，但首份最终 CSV 记 0；未发现
  能由公开测试解释的失败条件。
- 决策：评测镜像仅加入 hash 锁定的 NumPy 2.0.2；CPU rlimit 与配置的墙钟超时联动；
  `/99` 不为追平归档而人为判错，报告为 historical-label mismatch。主实验所有方法共享
  同一镜像、harness 和超时协议。

## A-017：历史 ActionNode 结构化填充

- 状态：`resolved-for-primary`
- 问题：历史 baseline 文件公开了 user prompt 和 Pydantic 输出字段，但实际格式指令由
  MetaGPT `ActionNode`/`code_fill` 注入；直接移植整个旧框架会同时带入模型、重试和成本逻辑。
- 决策：保留历史 user prompt，使用显式 system 消息要求等价的单字段 JSON；HumanEval/MBPP
  字段为完整 Python 程序。解析失败时保留原始响应并走确定性 fallback，不追加修复调用。
  该实现标记 `upstream-user/adapted-schema`，调用预算仍为 IO 每题 1 次。

## A-018：CoT-SC 的跨任务调用预算

- 状态：`resolved-for-primary`
- 现象：GSM8K/MATH/HumanEval/MBPP 历史实现为 5 个单调用候选 + 1 selector；HotpotQA
  的每个候选内部调用两次，实际为 11 次；DROP 实现缺失。
- 决策：主表按方法语义和公平预算统一为 5 个单调用 CoT 候选 + 1 selector（6 次）；
  HotpotQA 11 次只在 `archive-faithful` 模式保留。所有候选即使答案重复也必须真实独立请求，
  selector 解析失败记为该样本失败，不用 A 作为隐式默认值。

## A-019：MedPrompt 的 MATH 预算与重复候选映射

- 状态：`resolved-for-primary`
- 现象：论文/公开主结构为 3 候选 + 5 投票，但历史 MATH 入口执行 2 候选 + 2 投票；
  HotpotQA/DROP 缺失。历史 shuffle 又以 `solutions.index(solution)` 回查原始索引，候选文本
  重复时会全部映射到首次出现的位置。
- 决策：主表六任务统一 3+5（8 次请求）；按原始索引排列而非按文本回查；排列由固定 seed
  与 sample ID 决定；五票平局取最低原始索引。任一投票格式无效则样本失败，不丢弃无效票后
  用较小分母继续。历史 MATH 2+2 和重复文本 bug 仅在 `archive-faithful` 记录。

## A-020：MultiPersona 代码任务第二轮 context 丢失

- 状态：`resolved-for-primary`
- 现象：历史 HumanEval/MBPP 的第二轮调用为 `agent(problem, context, mode=...)`，但函数签名
  第二个位置参数是 `function_name`，真正的 `context` 仍为 `None`，所以三个角色实际上重复
  第一轮提示词；其他三个公开任务正确传入 peer context，DROP 实现缺失。
- 决策：主表统一采用方法定义所要求的两轮讨论，把三个角色第一轮 thinking 显式传入各自
  第二轮，仍严格保持 7 次调用；代码任务的历史错绑仅在 `archive-faithful` 模式保留。

## A-021：Self-Refine 的字典回填与末轮语义

- 状态：`resolved-for-primary`
- 现象：历史四任务把 `{"solution": ...}` 整个字典插入 review/revise prompt，而非只传答案
  文本；循环在第三次 review 为 false 时仍执行 revise，随后直接返回该未经复审的答案。
- 决策：主协议向 review/revise 显式传递 solution 文本；保持最多三轮和末轮 revise 后结束的
  调用图，以免改变论文预算。保存 `review_accepted`/`max_rounds_exhausted` 停止原因；字典
  `repr` 行为仅作为 `archive-faithful` 兼容项。

## A-022：AFlow 生成代码的宿主执行与 formatter 参数丢失

- 状态：`resolved-for-primary`
- 现象：论文同期 `Programmer`、`Test` 直接在宿主进程/进程池 `exec` 模型代码；其
  `CustomCodeGenerate._fill_node(..., function_name=...)` 又未把 function_name 传入
  `CodeFormatter`，entry point 约束实际丢失。
- 决策：保持 operator 名称、异步接口、prompt 和返回字段，但所有生成代码仅在锁定 Docker
  沙箱执行；代码生成提示词显式包含 entry point。安全修复不可在 `archive-faithful` 中关闭。

## A-023：官方最佳 workflow 的旧导入路径与可信边界

- 状态：`resolved-for-primary`
- 现象：官方 `graphs_test` 同时引用 `examples.aflow`、`metagpt.ext.aflow`，无法在独立 AFlow
  环境直接导入；动态加载 graph.py 还会执行归档中的任意顶层 Python。
- 决策：固定六个官方 best round 及 graph/prompt SHA256；主重放不动态执行归档代码，而由
  白名单原生 adapter 重建相同 operator 拓扑。prompt 仅允许 AST 字符串字面量赋值。结果
  标记 `official-best/native-safe-adapter`，与归档答案重评分分开报告。

## A-024：AFlow 优化器生成可执行 Python

- 状态：`controlled-adaptation`
- 现象：论文同期实现让 optimizer 直接生成并动态导入 Python graph；候选可执行任意顶层
  代码，且 graph 的静态合法性、内容去重和中断恢复都无法在执行前完整审计。
- 决策：optimizer 改为输出版本化 `declarative_v1` JSON graph，最多 10 个节点，只允许
  当前数据集在论文代码中声明的 operator、向后引用、字面量、列表和字符串拼接。候选 Python
  永不 import/exec；graph+prompt 内容哈希用于去重，完整候选哈希用于溯源。该变化保留
  operator/graph 搜索空间的核心语义，但明确标记为受控复现而非逐字节执行上游生成代码。

## A-025：ADAS 候选 `forward()` 与七个初始架构

- 状态：`protocol-compatible`
- 现象：官方 ADAS 让 meta-agent 生成 Python `forward(self, taskInfo)` 并在宿主 `exec`；
  seed 中还包含循环、提前停止与动态专家分支。AFlow Table 1 的六任务改写没有公开。
- 决策：保留七个 seed 的意图、完整 archive、每代 proposal+2 reflection 和 30 代搜索，
  但把架构表示为最多 12 节点的 `agent_dag_v1`。Self-Refine 最大 5 次修订静态展开，
  动态角色由 router 输出作为专家节点上下文；因此调用图不是原 Python 的 exact execution。
  所有候选只作数据解释，绝不在宿主或容器中执行 meta-agent 生成的控制代码。

## A-026：ADAS 失败代的调试与零分语义

- 状态：`resolved-for-primary`
- 现象：官方 ADAS 在候选执行异常或 validation 均为 0 时，可追加调试 meta prompt 并重试，
  失败后不加入 archive；本项目全局预注册协议要求最终失败进入错误报告并记 0。
- 决策：非法/重复 architecture 在加入 archive 前最多重新生成 3 次；合法 architecture 的
  validation 样本异常记 0 并保留在 fitness 分母，整代异常也以 0 分和错误文本进入 archive。
  这样不通过丢弃失败代改变实际生成代数，且所有成本与失败可审计。
