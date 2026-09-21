# 多传感器位姿融合与时间同步状态估计引擎 — 设计说明

本系统是一个**完全本地运行、零第三方依赖、确定性可复现**的状态估计引擎。它用本地
生成的传感器回放数据模拟惯性测量（IMU）、车体里程计（odometry）与绝对位姿
（absolute pose）三类观测，并系统性地处理乱序到达、丢帧、异常值、时钟漂移与长
时间运行问题。

所有时间单位为纳秒（`int64`），长度单位为米，角度单位为弧度。

---

## 1. 包结构

| 包 | 职责 |
|---|---|
| `pkg/linalg` | 自包含定长向量/矩阵：乘法、Cholesky、SPD 求解、对称 3×3 解析特征值 |
| `pkg/pose` | `Vec3`、Hamilton 单位四元数、`Pose(T_WB)`、`Twist` |
| `pkg/sensors` | 统一观测结构 `Observation` 与可区分的判定原因 `Reason` |
| `pkg/timesync` | 每路流的仿射时钟模型（偏移 + 一阶漂移） |
| `pkg/buffer` | 有界时间排序接入缓冲（重排序窗口、水印、重复/迟到判定） |
| `pkg/outlier` | 新息马氏距离门限（卡方分位，Wilson–Hilferty 近似） |
| `pkg/fusion` | ESKF：9 维误差状态（位置/速度/姿态），四元数姿态，Joseph 协方差 |
| `pkg/replay` | 固定种子的确定性数据生成（乱序/丢帧/异常值/漂移） |
| `pkg/engine` | 并发串行化、时钟配对、降级策略、规模/耗时上限、决策审计 |
| `cmd/fusiondemo` | 端到端演示入口 |
| `test/integration` | 乱序、丢帧、异常值、漂移、长时运行、并发的端到端测试 |

---

## 2. 状态与坐标系约定

### 2.1 坐标系

- **世界系 W**：固定的右手系；重力取 `g_W = (0, 0, -9.81) m/s²`。
- **机体系 B**：与平台固连。
- **位姿 `T_WB`**：`p_W`（位置）与 `q_WB`（姿态），把机体系向量旋转到世界系：
  `v_W = q_WB ⊗ v_B ⊗ q_WB*`。
- 里程计速度在机体系给出，更新前由 `R_WB` 旋转到世界系，使观测矩阵保持简单
  的选择阵，杜绝机体/世界雅可比混用。

### 2.2 四元数约定

- Hamilton 约定，分量顺序 `(w, x, y, z)`。
- 每次传播与每次修正后都做**归一化**，并固定标量分量非负的半球，保证长时间
  积分下姿态连续、不会出现 `q → -q` 的跳变。
- 退化输入（零四元数、零轴、非有限值）**确定性地**映射为单位四元数，绝不产生
  NaN。

### 2.3 滤波器状态

ESKF 名义状态：`p_W (3), v_W (3), q_WB`；误差状态 9 维：

```
δx = [ δp (3), δv (3), δθ_W (3) ]
```

姿态误差约定为 `q_true = exp(δθ_W) ⊗ q_est`，因此姿态新息与修正都在世界系。

- **预测**：`p ← p + v dt + ½ a_W dt²`，`v ← v + a_W dt`，
  `q ← q ⊗ exp(ω_B dt)`；其中 `a_W = R_WB · f_B + g_W`。
  缺测（`lastIMU == nil`）时按零角速度、零世界系加速度处理（等价"假设静止"）。
- **过程噪声**：白加速度噪声（PSD `σ_a²`）与白陀螺噪声（`σ_g²`），离散化
  `Q` 含 `σ_a²·dt³/3`（位置）、`σ_a²·dt²/2`（耦合）、`σ_a²·dt`（速度）、
  `σ_g²·dt`（姿态），协方差通过 `F P Fᵀ + Q` 传播。
- **观测更新**（Joseph 稳定形式）：

```
S = H P Hᵀ + R
K = P Hᵀ S⁻¹          (S 由 Cholesky 求解，非 SPD 直接拒绝)
P ← (I−KH) P (I−KH)ᵀ + K R Kᵀ
```

### 2.4 数值稳定性保证

1. 每次更新后 `P = sym(P)`；
2. 协方差使用 Joseph 形式（对增益误差不敏感）；
3. 每次更新/传播后执行 **SPD 修复**：Cholesky 失败时按 10 倍递进加对角脊
   （初始为 `MinEigenvalue`，默认 1e-12），40 次仍失败则回退为保守对角阵——
   协方差**永远**正定；
4. 所有线性求解走 Cholesky 并显式检查，退化新息协方差被判定为
   "degenerate innovation covariance" 而不会静默污染状态；
5. 绝对位姿更新额外有位置修正量硬界 `MaxPositionJump`。

---

## 3. 时间同步规则

### 3.1 每路流一个仿射时钟模型

```
localNS = (1 + drift) · nativeNS + offsetNS
```

- `drift` 无量纲（如 8 ppm = 8e-6），`offsetNS` 为纳秒偏移。
- 用配对样本 `(nativeNS, 到达时刻参考时 − 已知固定链路延迟)` 做中心化最小二乘
  拟合；计算只依赖样本，**无随机量、无墙钟参与**，同输入同结果。
- 仅接受 **native 时间严格单调递增**的配对样本：乱序到达的迟到帧不会被用于
  拟合（否则两帧顺序颠倒会算出巨大的虚假斜率）。
- 拟合漂移超出 `MaxDrift` 硬界（默认 1e-3）时，模型拒绝就绪，而不是用荒谬的
  参数扭曲时间。
- `MinPairs`（默认 2）个样本前模型 `Ready=false`，映射 `ToLocal` 失败。

### 3.2 已知固定链路延迟

观测带 `ArrivalRefNS`（确定性回放中 = 测量时刻 + 固定 `LinkDelay`，乱序帧再
附加重排延迟）。引擎在配对前从到达时刻中**减去配置的固定链路延迟**，使恒定
传输延迟不被误判成时钟偏移——否则偏移会与漂移项耦合，在长时运行中产生
`offset·drift` 量级的时间偏差（这是开发中实测发现并修正的问题）。

### 3.3 时钟未就绪的确定性行为

- 每路流前 `MinPairs-1` 个观测**不参与状态推进**，返回
  `clock_model_not_ready`；它们也不进入缓冲、不占用水印。
- 时钟就绪后该路观测按校正后的 `LocalNS` 排序与消费。
- 判定原因与门限拒绝、迟到丢弃等完全可区分。

---

## 4. 乱序、重复与倒退时间戳

### 4.1 重排序窗口与水印（`pkg/buffer`）

- 观测按 `LocalNS` 升序保存；同时间戳的平局按 `(stream, seq, 到达序号)`
  **完全确定性**排序。
- **释放规则**：本地时间 `t` 的观测在 `now ≥ t + WindowNS` 后才可释放（重排序
  延迟，默认 50–80 ms），给迟到帧留出按时间归位的窗口。
- **水印 watermark**：已释放的最大时间戳。

| 情形 | 结果 | 判定原因 |
|---|---|---|
| 顺序到达 | 入缓冲，按序释放 | `ok` |
| 迟到但仍早于水印前（在窗口内） | 按时间归位，正确参与推进 | `out_of_order_accepted` |
| 同源同 `(stream,nativeNS)` 重复 | 拒绝，不消费 | `duplicate_timestamp` |
| 时间戳早于已释放水印 | 拒绝，**不静默丢弃** | `dropped_late_behind_watermark` |
| 早于引擎启动时间 | 拒绝 | `before_engine_start` |
| 缓冲满 | 拒绝，无部分写入 | `buffer_backpressure` |

关键不变量：**状态只沿时间前进**；任何观测都不会被提前消费，也不会被静默
丢弃——每次拒绝都带原因与明细（含时间戳、水印等判定依据）。

### 4.2 时间倒退的滤波保护

`Filter.PredictTo` 拒绝倒退时间（返回 false，状态不变），返回
`timestamp_regression`。

---

## 5. 异常值判定策略

更新前对新息 `innov` 与新息协方差 `S` 做可信度检验：

```
d² = innovᵀ S⁻¹ innov
```

- 门限为 `dof` 维卡方分布在配置 σ 分位（默认 4σ）处的分位数，用确定性的
  Wilson–Hilferty 立方根近似计算（dof≥1 精度约 0.02），无需查表/第三方库。
- 可选的 `MaxChi` 对新息各分量做绝对硬截止。
- 三种可区分结论：
  - `accept`：应用更新；
  - `mahalanobis_gate_reject`：超门限，明细含 `d²`、阈值、σ、dof；
  - `degenerate innovation covariance`：`S` 非正定，明细明确标注。
- 另有位置修正硬界 `MaxPositionJump`（超界即拒）。
- **被拒观测不修改状态与协方差**（测试断言拒绝前后 `Pos` 字节一致）。
- 回放每 11 个绝对位姿注入一个 +8 m / +0.6 rad 的异常值，集成测试断言这些异常
  值全部被门限拦截（实测约 5–9 次/20 s，取决于门限）。

---

## 6. 缺测与传感器降级

每路流独立维护健康状态（`engine.StreamHealth`）：

| 模式 | 条件 | 行为 |
|---|---|---|
| `unavailable` | 从未产生过可用样本 | 不参与更新 |
| `ok` | 距上次样本 `age ≤ DegradeAgeNS` | 正常更新 |
| `degraded` | `DegradeAgeNS < age ≤ StaleAgeNS` | **接受该帧，但测量 σ 乘以 `SigmaFactor`**（不确定度膨胀），决策原因 `stream_degraded` |
| `stale` | `age > StaleAgeNS` | 决策标记 `stream_stale`；陈旧观测永远不会被当作最新数据 |

- 降级判断针对"上一帧到这一帧的缺口"，恢复后的第一帧自带原因标记。
- `AdvanceTo` 每次基于当前时间刷新所有流的模式，缺测路在健康报告中显式可见。
- 从不读取或重放陈旧观测：参与更新的永远是当前被消费的时间戳对应的样本。

---

## 7. 并发一致性

- 引擎用**单一互斥锁**串行化 `Submit / AdvanceTo / Snapshot / Health /
  AcceptedCount / Close`；快照返回防御性拷贝。
- 预测、更新、快照交错时不可能观察到"半更新"状态（位置/姿态/协方差/健康/
  计数在同一个临界区内一起提交）。
- `-race` 下的测试持续并发快照，断言四元数恒为单位、协方差恒为 SPD。
- **可复现性边界**：相同的提交序列（同顺序、同时刻）重复运行得到逐位一致的
  状态与计数（`TestReplayReproducible`）。不同 goroutine 交错构成*不同的输入
  序列*——某次交错中水印先推进、迟到帧后到，则该帧被确定性丢弃，这是正确且
  可解释的行为，不承诺跨不同交错顺序的逐位一致。

---

## 8. 规模与耗时上限

`engine.Limits`：

- `MaxPending`：缓冲在途观测上限，超限返回 `buffer_backpressure`；
- `MaxObservations`：提交尝试总数（接受+拒绝）上限，超限返回
  `size_limit_exceeded`——拒绝也计数，调用方无法靠重试让状态无限增长；
- `StepBudget`：单次 `Submit/AdvanceTo` 的墙钟处理预算。

耗时预算的实现保证**无部分更新、无资源泄漏**：

- 仅在完整处理边界检查预算；预算耗尽时，**尚未开始**的观测原样留在有序缓冲中
  （`PopReady` 在不可释放时不改变状态），下次 `AdvanceTo` 继续；
- 计时用可注入的单调时钟函数（默认墙钟），但**处理顺序不依赖计时**；
- `Close` 释放缓冲并拒绝后续所有提交。

---

## 9. 确定性回放（`pkg/replay`）

- 固定种子（默认 20260921）的 splitmix64 + Box–Muller 高斯噪声；
- 解析真值轨迹（匀速直线 + 正弦侧向运动 + 平滑组合轴转动），任意时刻真值无
  插值误差；
- 可配置：采样率、每流时钟偏移与 ppm 漂移、固定链路延迟、每 k 帧乱序延迟、
  每 n 帧丢帧、每 n 个位姿注入异常值、各噪声 σ；
- 交付顺序按 `ArrivalRefNS`（平局按 stream、id）排序——同种子两次 `Generate`
  得到逐字段相同的序列。

默认场景（20 s）：IMU 200 Hz、里程计 20 Hz、绝对位姿 5 Hz；每 17 帧一次
120 ms 乱序延迟；每 23 帧丢一帧；每 11 个位姿注入一个异常值。

---

## 10. 判定原因速查（`sensors.Reason`）

| 原因 | 含义 |
|---|---|
| `ok` | 正常接受并消费 |
| `out_of_order_accepted` | 迟到但在水印前，已按时间归位 |
| `dropped_late_behind_watermark` | 晚于已释放水印，确定性丢弃 |
| `duplicate_timestamp` | 同源同时间戳重复 |
| `timestamp_regression` | 滤波时间倒退被拒 |
| `clock_model_not_ready` | 时钟配对样本不足 |
| `before_engine_start` | 早于启动时间 |
| `mahalanobis_gate_reject` | 马氏门限拒绝 / `S` 退化 / 位置硬界 |
| `stream_stale` / `stream_degraded` | 缺测降级 / 陈旧 |
| `unsupported_observation` | 未知流或载荷/噪声参数缺失 |
| `size_limit_exceeded` | 总规模上限 |
| `buffer_backpressure` | 缓冲/在途上限 |
| `processing_budget_exceeded` | 单次处理耗时预算耗尽（观测保留） |
| `stream_inactive` | 流始终不可用 |

---

## 11. 验证方法

```bash
# 全部单元测试（数值正确性、门限、缓冲、时钟、滤波器收敛/SPD）
go test ./... -count=1

# 含数据竞争检测（并发一致性）
go test ./... -race -count=1

# 含 10 分钟长时运行（默认跳过）
go test ./test/integration/ -count=1

# 端到端演示（打印判定原因分布、健康状态、真值误差）
go run ./cmd/fusiondemo -samples
go run ./cmd/fusiondemo -duration 60s -seed 1 -verbose
```

典型 20 s 回放结果：约 4300 条观测，平均位置误差 ~1.6 cm、最大 ~3.4 cm，
平均姿态误差 ~0.012 rad；门限拒绝全部注入异常值；四元数范数
`1.000000000000`、协方差 Cholesky 始终通过。10 分钟 / 12 万+ 次更新后位置
误差仍 ~2 cm，无姿态退化、无协方差失正定。

### 测试覆盖矩阵

| 需求 | 测试 |
|---|---|
| 乱序按序参与 | `buffer.TestOutOfOrderDrain`、`engine.TestDuplicateAndLateAndOrder` |
| 重复/倒退可区分 | `buffer` 重复/迟到用例、`fusion.TestPredictRegressionRejected` |
| 旋转数值稳定 | `pose.TestLongChainStaysUnit`、`fusion.TestCovarianceStaysSPD` |
| 异常值不污染 | `fusion.TestGateRejectDoesNotPollute`、`engine.TestGateRejectionAudit` |
| 时钟漂移恢复 | `timesync.TestClockRecovery`、集成回放 |
| 缺测/降级 | `engine.TestDegradationAndStale` |
| 并发一致/无竞争 | `engine.TestReproducibleUnderConcurrency`、`integration/TestConcurrentSubmissionsNoRace` |
| 规模/耗时上限 | `engine.TestSizeLimit`、`engine.TestProcessingBudgetNoPartial` |
| 长时运行 | `integration/TestLongRunStability`（10 min） |
| 可复现 | `replay.TestDeterministicGeneration`、`integration/TestReplayReproducible` |
