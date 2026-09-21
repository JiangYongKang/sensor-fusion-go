# sensor-fusion-go

本地运行的**多传感器位姿融合与时间同步状态估计引擎**。用确定性回放数据模拟
惯性测量（IMU）、里程计与绝对位姿观测，在乱序到达、丢帧、异常值、时钟漂移与
长时间运行下给出**一致、可解释、可复现**的位姿估计。

- 零第三方依赖（线性代数、卡方分位、随机数全部自包含）。
- ESKF（位置/速度/姿态 9 维误差状态）+ Hamilton 单位四元数 + Joseph 协方差
  更新 + 自适应 SPD 脊修复。
- 每路流仿射时钟模型（偏移 + ppm 漂移），有界重排序窗口与水印语义，重复/倒退/
  迟到全部给出可区分原因，绝不静默丢弃或提前消费。
- 马氏距离门限 + 位置修正硬界；异常观测不污染状态。
- 缺测降级（不确定度膨胀 / 陈旧标记），陈旧观测不参与更新。
- 单锁串行化，`-race` 安全；相同输入序列逐位可复现。
- 可配置在途规模、总量规模与单次处理耗时预算；预算耗尽无部分更新、无泄漏。

## 快速开始

```bash
go run ./cmd/fusiondemo -samples          # 20s 确定性回放 + 决策样例
go run ./cmd/fusiondemo -duration 60s -seed 1 -verbose
```

示例输出（20 s 默认场景）：

```
observations=4306 accepted=4030 rejected=276
  decision out_of_order_accepted            53
  decision dropped_late_behind_watermark    248
  decision clock_model_not_ready            23
  decision mahalanobis_gate_reject          5
  decision stream_degraded                  4
truth error: meanPos=0.0164m maxPos=0.0307m meanAtt=0.01223rad
final: |q|=1.000000000000 covSPD=true
```

## 测试

```bash
go test ./... -count=1                 # 全部单元测试
go test ./... -race -count=1           # 含数据竞争检测
go test ./test/integration/ -count=1   # 含 10 分钟长时稳定性测试
```

测试重点为乱序、丢帧、异常值、时钟漂移、长时间运行与并发，覆盖成功与失败
路径；关键输入与判定依据（时间戳、水印、`d²`、阈值、σ、自由度等）均通过
`t.Logf` 打印。

## 文档

完整的状态/坐标系约定、时间同步规则、异常判定策略、降级语义、并发与资源
边界、判定原因速查与验证方法见 [`docs/DESIGN.md`](docs/DESIGN.md)。

## 包概览

```
pkg/linalg    自包含小矩阵（Cholesky / SPD 求解 / 3x3 解析特征值）
pkg/pose      Vec3 / 单位四元数 / T_WB 位姿
pkg/sensors   统一观测类型与可区分判定原因
pkg/timesync  仿射时钟模型（偏移 + 漂移，最小二乘）
pkg/buffer    有界乱序缓冲、水印、重复/迟到处置
pkg/outlier   马氏距离门限（卡方 / Wilson-Hilferty）
pkg/fusion    ESKF 状态预测与观测更新
pkg/replay    固定种子确定性数据生成
pkg/engine    并发、降级、规模/耗时上限、审计
cmd/fusiondemo 端到端演示
```
