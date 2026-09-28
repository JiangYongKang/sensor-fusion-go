# sensor-fusion-go：多传感器位姿融合与时间同步状态估计引擎

本地运行的多传感器（IMU / 里程计 / 绝对位姿）位姿融合引擎，
以本地生成的确定性回放数据验证乱序、丢帧、异常值、时钟漂移与
长时间运行场景下估计的一致性、可解释性与可复现性。

## 状态与坐标系约定

- **世界系**：ENU（东-北-天），重力加速度 `g = (0, 0, -9.81)`。
- **名义状态** `State`：位置 `p`、速度 `v`（世界系）、姿态单位四元数
  `q`（机体系→世界系，Hamilton 约定，`w` 为标量部）。
- **误差状态**（9 维）：`[δp, δv, δθ]`，其中 `δθ` 为机体系旋转向量，
  通过右乘更新 `q ← q ⊗ exp(δθ)`，每步归一化。
- **IMU 观测**：比力 `a_b`（机体系，含重力反作用）与角速度 `ω_b`。
  预测模型：`p ← p + v·dt + ½(R·a_b + g)dt²`，`v ← v + (R·a_b + g)dt`，
  `q ← q ⊗ exp(ω_b·dt)`。无 IMU 时退化为恒定速度模型（不引入重力）。
- **里程计观测**：机体系线速度，观测模型 `h = Rᵀ·v`。
- **绝对位姿观测**：世界系位置与姿态，姿态残差 `log(q̂⁻¹ ⊗ q_obs)`。
- **协方差**：9×9 误差状态协方差，预测用 `F·P·Fᵀ`，更新用 Joseph
  形式 `(I−KH)P(I−KH)ᵀ + KRKᵀ`，每步对称化并对对角元钳制下界，
  保证长时间运行不失去正定性。

## 时间同步规则

- 所有观测携带传感器时间戳 `Stamp` 与单调序号 `Seq`，
  全序定义为 `(Stamp, Seq, Kind)`，保证排序确定无歧义。
- **乱序**：摄入分两阶段——先入重排缓冲，再按全序出队消费。
  `ReorderDelay > 0` 时，时间戳晚于 `最新观测 − ReorderDelay` 的观测
  留在缓冲中等待乱序伙伴；为 0 时每批摄入后立即按序消费。
- **重复**：与已消费观测或缓冲内待定观测 `(Kind, Stamp, Seq)` 完全
  相同的观测以 `duplicate` 拒绝，绝不重复参与更新。
- **倒退**：时间戳早于已消费序列的观测以 `timestamp_regress` 拒绝；
  落后超过 `StaleAfter` 的以 `stale` 拒绝，绝不把陈旧数据当最新数据。
- **时钟漂移（在线对齐，新增能力）**：`EnableAlignment=true` 时为每路
  传感器在线估计其时钟到统一时间轴的仿射关系
  `unified = skew·raw + offset`，换算后的时间真正参与状态推进、顺序
  判定、陈旧/倒退判定与降级判定。
  - 以一路（默认 IMU，可用 `AlignAutoRef/AlignRef` 选择）为时间基准，
    其余为待对齐路。
  - 利用各路共享的机身偏航运动做跨路时延估计：信号按时间有序缓存，
    连续偏航角由有序环统一积分/解卷绕（免疫批内乱序），在居中窗口上
    做去均值形状相关 + 联合 (skew, lag) 网格搜索得到锚点；锚点用分段
    线性模型换算，长历史稳健跟踪斜率，近期窗口决定时延。
  - **状态机（原因可区分，绝不把未对齐数据当已对齐）**：
    `cold`（无漂移证据，恒等放行、零额外开销，因此关闭对齐或无漂移
    输入的既有行为完全不变）→ `calibrating/align_calibrating`
    （样本不足/参考未覆盖，观测暂缓不入更新）→ `locked`（锚点数、跨度、
    残差、斜率/偏移范围均达标，按换算时间消费）；斜率或偏移超
    `AlignMaxSkew/AlignMaxDrift` 判 `out_of_range / time_unavailable`。
  - **校准宽限**：某路首次出现后 `AlignCalibrateAfter`（默认 5s）内
    暂不采用；超过宽限仍未锁定，则按原始时间（恒等）**降级放行**参与
    融合并标记降级，保证对齐层永不饿死滤波器（只有真正越界才持续拒绝）。
  - 一路长时间无数据时，对齐水线在 `DegradeAfter+AlignSearchRange`
    后不再被该路冻结，避免单路缺测阻塞其他路、撑爆缓冲。
  - 零漂移输入下估计恒为 skew=1、offset=0（整数纳秒坐标），结果与
    原始时间戳行为逐位一致；相同输入重复运行严格一致（对齐相关为
    纯确定性计算，不读墙钟、无随机数）。
  - 配置：`AlignDetect / AlignMinSamples / AlignMinSpan / AlignTolerance
    / AlignMaxSkew / AlignMaxDrift / AlignWindow / AlignSearchRange /
    AlignReleaseMargin / AlignCalibrateAfter / AlignAutoRef / AlignRef`。
  - 查询：`Engine.AlignmentStats()` 返回每路状态/skew/offset/残差/样本；
    `Engine.AlignedTime(kind, raw)` 返回某路原始戳对应的统一时间。
- 相同输入序列重复运行得到一致结果（引擎内部无随机性；唯一的耗时
  上限检查按既定策略拒绝并记录，不产生部分更新）。

## 异常判定与降级策略

- **门限剔除**：每次更新计算马氏平方 `d² = rᵀS⁻¹r`，超过 `GateChi2`
  （默认 16，约 6 维 3σ）的观测以 `outlier_gate` 拒绝，状态与协方差
  不被污染（判定发生在预测推进之后、误差注入之前）。
- **缺测降级**：某路观测间隔超过 `DegradeAfter` 即标记降级；降级期间
  到达的观测不直接拒绝，而是将观测噪声膨胀 `DegradedNoiseInflate`
  倍（默认 100）后参与更新，判定原因为 `sensor_degraded`，
  既避免陈旧数据主导估计，又允许传感器平滑恢复。
- **初始化**：收满 `InitWindow` 条且含绝对位姿时初始化；无绝对位姿
  时累积 3 倍窗口后以零位姿兜底启动。
- **规模与处理预算（边界策略明确、拒绝后引擎继续）**：
  - 批大小超 `MaxBatchSize`：整批 `limit_exceeded`，无部分更新；
  - 待处理积压超 `MaxPendingObs`：该条 `backlog_exceeded`；参考路
    （时间基准）不被积压拒绝，避免缺测窗口把统一时间轴一起饿死；
  - 确定性操作预算 `MaxBatchOps`：按对齐相关折算的操作数在批前预估，
    剩余预算不足以完成本批时整批 `process_budget`（原因与批大小、积压
    超限区分开）；`MaxProcessTime` 仅作宽松墙钟兜底。对齐层失败时
    整批回滚（不破坏已建立的时钟基准与各路模型）。
  - 所有拒绝均不留部分更新，拒绝后引擎仍可继续处理后续合法数据；
    待处理数据在运行期始终有界。
- 每次判定（接受/拒绝及原因、马氏距离）写入有界判定日志，
  可通过 `Decisions()` 获取，原因枚举可区分：
  `accepted / outlier_gate / stale / duplicate / timestamp_regress /
  insufficient_data / align_calibrating / time_unavailable /
  sensor_degraded / limit_exceeded / backlog_exceeded / process_budget`。
  其中“样本不足暂不可用”(`align_calibrating`) 与“漂移超界”
  (`time_unavailable`) 明确区分。

## 并发一致性

- 引擎全部状态变更在写锁内原子完成，快照读取走读锁；
  预测、更新、快照交错不会出现半更新状态（`-race` 测试覆盖）。
- `Snapshot()` 返回深拷贝，调用方持有期间引擎可继续推进。

## 回放数据与验证方法

- `GenerateReplay(ReplayConfig)` 生成确定性多路回放。真值轨迹有两种：
  默认的水平圆周 + 垂直正弦（恒定速率），以及 `Excited=true` 的**时变
  速率**轨迹（偏航角速率随时间正弦变化，供在线时间对齐获得可辨识的
  跨路运动信号）。叠加高斯噪声、丢帧、异常值、局部乱序、线性时钟漂移
  （`ClockDrift`）、**各路独立的分段/方向可变时钟漂移曲线**
  （`Clocks`：每路可设多段斜率与固定偏置，模拟时钟忽快忽慢）、
  绝对位姿缺测（`PoseGap`）与任意一路长时间缺测（`SensorGap`）。
  相同 `Seed` 生成完全相同序列。
- 测试覆盖（`fusion/` 包内）：
  - 四元数/矩阵数值稳定性（近零、大数、10 万步积分不退化）；
  - 乱序与顺序摄入殊途同归（`TestOutOfOrderDeterministic`、
    `TestReorderDelayWindow`）；
  - 重复/倒退/陈旧/漂移的可区分拒绝；
  - 异常值门限拒绝且不污染状态（与参照运行逐位比较）；
  - **在线时间对齐**：时变分段漂移下两路均锁定、换算后时间参与融合
    且时间错位很小（`TestAlignmentTracksTimeVaryingDrift`）；
    恒定慢漂移斜率跟踪（`TestAlignmentConstantSkew`）；样本不足判
    `align_calibrating`（`TestAlignmentUnavailableReason`）；斜率越界
    判 `out_of_range/time_unavailable`（`TestAlignmentOutOfRange`）；
  - **容量与处理预算边界**：单批超限 `limit_exceeded`、积压过多
    `backlog_exceeded`、确定性操作预算不足 `process_budget`，且拒绝
    后引擎继续（`TestBatchLimitRejected/TestBacklogRejected/
    TestDeterministicOpBudget`）；
  - 缺测降级与噪声膨胀恢复；
  - 预测/更新/快照并发交错一致性（`-race`，
    `TestConcurrentAlignmentConsistency`）；
  - 60 秒全场景（`TestLongRunFusion`）与 **30 分钟**大流量长跑
    （`TestLongRunHalfHour`，>22 万观测）：姿态/速度/协方差稳定、
    四元数单位、待处理积压有界、离群被捕获、重复运行严格一致。
- 运行验证：

```sh
go test ./... -race -count=1 -v   # 全部测试（日志含关键输入与判定依据）
go run ./cmd/demo                 # 端到端回放演示
```
