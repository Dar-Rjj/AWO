# Table 1 pilot 与 full 预算门禁

生成时间：2026-08-30。完整 Table 1 **尚未启动**；本文档是启动前的费用确认依据。

## Pilot 范围与完整性

- 配置：`configs/pilot.yaml`，6 个数据集 × 8 个方法，每个 validation/test split 取 20 条，测试重复 1 次。
- 编排阶段：30/30 个 summary；搜索最佳候选：12/12；结果单元：48/48；测试记录：960/960。
- 所有 12,378 个正式请求的 requested/actual model 均为 `deepseek/deepseek-chat`，状态均为 `ok`。
- 实际路由：DeepInfra 5,600 次、Novita 3,378 次、StreamLake 3,400 次。
- 账本重放前后均为 12,378 条请求，组合 SHA256 均为
  `f9b2345320340612acf4454d986a1886a9795087a92cb477d61d208c239eb12d`，未产生重复调用。
- 完整 48 单元分数表和逐 run 审计位于本地生成目录
  `reports/generated/table1-pilot/report.md` 与 `report.json`；该目录按设计不提交大体积生成物。

## Pilot 实测资源

| 阶段 | 请求 | Tokens | 费用（USD） |
| --- | ---: | ---: | ---: |
| ADAS + AFlow validation search | 8,239 | 5,908,079 | $2.835821 |
| 六个 baseline + 冻结 ADAS/AFlow test | 4,139 | 3,243,573 | $1.491862 |
| 合计 | 12,378 | 9,151,652 | $4.327683 |

以上不包含独立连通性 preflight（1 次请求、28 tokens、$0.0000134）。Pilot 平均成本为
$0.00034963/请求，或 $0.47289/百万 tokens；这两个均是该次动态 provider 路由下的观测值。

## Full 分层外推

中心估计按数据集分别缩放，而不是对总成本使用单一倍数：

- 所有 test run：乘以 `full_test_count × 3 / 20`。
- ADAS search：pilot 为 7 个初始 architecture + 3 个生成轮；full 为 7 + 30，按
  `(37 / 10) × (full_validation_count / 20)` 缩放。
- AFlow search：pilot 为 1 个初始 workflow + 3 个生成轮、每轮 validation 重复 2 次；
  full 为 1 + 20、重复 5 次，按 `(21 / 4) × (5 / 2) × (full_validation_count / 20)` 缩放。
- 假设每个数据集/方法的平均 workflow 节点数、输出长度、失败修复率和单 token 价格保持
  与 pilot 相近；搜索轨迹变化会打破这一假设。

| Full 阶段 | 估计请求 | 估计 Tokens | 估计费用（USD） |
| --- | ---: | ---: | ---: |
| 全部 test（8 方法，3 repeats） | 387,552 | 317,308,154 | $138.62 |
| ADAS search（30 生成轮） | 160,786 | 122,642,118 | $52.33 |
| AFlow search（20 生成轮 × 5 validation repeats） | 253,173 | 181,987,430 | $100.19 |
| 中心估计合计 | 801,511 | 621,937,702 | **$291.15** |

编排器静态逻辑区间为 421,397–1,827,506 次请求。仅将 pilot 平均单请求成本套入该区间，
得到 **$147.33–$638.95** 的粗略费用参考范围；上界按 AFlow 10 节点、ADAS 12 节点和
最大候选生成尝试数计算，较为保守，但它仍不覆盖未来价格变化或异常重试风暴。

## 启动门禁

完整实验命令继续受 `--execute --confirm-full-table1` 双重开关保护。本阶段只完成 pilot、
聚合、账本重放和预算估计；在用户再次确认 full 预算前，不运行完整 Table 1。
