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
- **时钟漂移（在线对齐）**：`EnableAlignment=true` 时为每路传感器在
  线估计其时钟到统一时间轴的映射，换算后的统一时间**真正、且唯一**
  参与状态推进、顺序判定、陈旧/倒退判定、降级判定与消费水线计算。
  - 以一路（默认自动选 IMU，可用 `AlignAutoRef/AlignRef` 指定）为
    **时间基准（参考路）**，参考路恒等映射；统一时间轴即参考路时钟。
  - 利用各路共享的机身偏航运动做跨路时延估计：信号按时间有序缓存，
    连续偏航角由有序环统一积分/解卷绕（免疫批内乱序），在滞后居中窗
    口上做去均值形状相关 + 联合 (skew, lag) 网格搜索得到锚点；锚点用
    **分段线性换算**（天然跟随分段换向的时变漂移），长历史稳健估计
    斜率、近期锚点决定时延。`verdict / Remap / AlignedTime / 水线`
    全部走同一个换算入口 `mapTimeNS`，保证“对外显示的时间”与“实际
    参与状态推进的时间”严格一致。
  - **状态机（显示什么口径，就一定按什么口径消费）**：
    - `cold`：尚无统计显著的漂移证据，按原始戳**恒等放行**。判定漂移
      证据时不看原点截距（恒定斜率漂移在原点截距≈0），而要求近期锚点
      直接测得的 lag（y−x）中位数超过 `AlignDetect`，因此陀螺噪声积分
      造成的毫秒级假峰不会误触发。零漂移输入全程停在 cold。
    - `calibrating`：已有漂移证据但锚点数/跨度/残差尚不可信。观测
      **暂缓、绝不参与更新**，对外原因 `align_calibrating`（=样本不足/
      参考未覆盖，可与“超范围”明确区分）。
    - `locked`：锚点数、跨度、残差、斜率/偏移范围全部达标，按换算时间
      消费；`AlignedTime` 仅在此态返回 `(t,true)`。
    - `out_of_range`：斜率或偏移超 `AlignMaxSkew/AlignMaxDrift`，观测
      以 `time_unavailable` 拒绝（与样本不足的 `align_calibrating`
      分开），且**不进入待处理缓冲**。
  - **校准宽限（不饿死、不冻结、不堆积）**：宽限自“发现漂移证据”起算
    `AlignCalibrateAfter`（默认 5s，而非自该路首次出现起算）。宽限内
    观测暂缓；超过宽限仍未锁定则按原始戳**恒等降级放行**并把观测噪声
    膨胀（`sensor_degraded`），只有真正越界才持续拒绝。
  - **统一时间轴不被单路卡住**：对齐水线只取参考路与**已锁定**路的覆盖
    交集；`cold / calibrating / out_of_range` 路不冻结水线，参考路与
    其他路照常推进。已锁定路超过 `DegradeAfter+AlignSearchRange` 缺测
    时按暂时缺测处理。永久落后于水线的暂缓项（越界旧积压等）会被显式
    驱逐并记 `time_unavailable/stale`，因此**待处理数据在运行中始终
    有界**，不会因一路对不齐而无限堆积。
  - **旧量测护栏**：无论因锁定后批量重判、映射切换还是迟到到达，凡是
    换算时间早于当前状态超过 `StaleAfter` 的观测，一律在消费侧以
    `stale` 拒绝，绝不把几秒前的位姿/速度当成“当前”量测注入——这是
    时变漂移下不发散的关键。
  - **零漂移兼容**：无漂移输入各路恒为 cold、恒等映射，开对齐与关对齐
    的位姿/速度结果**逐位一致**；相同输入重复运行严格一致（对齐相关
    为纯确定性计算，不读墙钟、无随机数）。
  - 配置：`AlignDetect / AlignMinSamples / AlignMinSpan / AlignTolerance
    / AlignMaxSkew / AlignMaxDrift / AlignWindow / AlignSearchRange /
    AlignReleaseMargin / AlignCalibrateAfter / AlignAutoRef / AlignRef`。
  - 查询：`Engine.AlignmentStats()` 返回每路状态/skew/offset/残差/样本；
    `Engine.AlignedTime(kind, raw)` 仅在 locked 时返回 `(统一时间,true)`。
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
  - **对齐相关的待处理边界**：校准宽限内暂缓项只可能来自最近
    `AlignCalibrateAfter` 窗口（有界）；`out_of_range` 新观测直接拒绝、
    不入缓冲；锁定后转越界/永久落后水线的旧暂缓项在排水时被显式驱逐
    （记 `time_unavailable/stale`）。因此待处理规模始终 ≤
    `MaxPendingObs`，运行中不随时间线性增长。
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
  - **在线时间对齐**：时变分段换向漂移下，直接断言**开对齐的融合误差
    不劣于关对齐、且全程有界不发散**，两路都真正锁定
    （`TestTimeVaryingDriftAlignedNotWorse`，1 分钟与 3 分钟逐点对比，
    不放宽噪声或既有精度门限）；零漂移输入开/关对齐**逐位一致**且各路
    保持 `cold`（`TestZeroDriftAlignmentIsBitIdentical`）；对外换算时间
    就是消费口径（`TestAlignmentDisplayedTimeIsConsumedTime`）；样本不足
    暂缓且积压有界、宽限后不饿死（`TestAlignmentPendingBoundedWhen
    Calibrating`）；越界判 `out_of_range/time_unavailable` 且**不堆积**
    （`TestAlignmentOutOfRangeRejectedNotBuffered`、
    `TestAlignmentOutOfRange`）；一路不可信不冻结时间轴
    （`TestAlignmentDoesNotFreezeTimeline`）；恒定慢漂移斜率跟踪
    （`TestAlignmentConstantSkew`）；
  - **30 分钟长跑（真实叠加各路时变漂移）**：开对齐、激励轨迹、各路多
    段换向漂移 + 离群/缺帧/乱序/一路长缺测下，误差有界不发散、两路全程
    锁定、积压有界、重复运行严格一致（`TestLongRunHalfHourAligned`，
    >22 万观测）；关对齐基线 `TestLongRunHalfHour` 保留不回归；
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
- 运行验证（在仓库根目录执行）：

```sh
# 快速逻辑回归（-short 会跳过两个 30 分钟长跑）
go test ./... -short -count=1

# 本次修复的核心场景（时变漂移开对齐不劣于关对齐、不发散）
go test ./fusion -count=1 -v -run TestTimeVaryingDriftAlignedNotWorse
# 零漂移开/关对齐逐位一致
go test ./fusion -count=1 -v -run TestZeroDriftAlignmentIsBitIdentical
# 样本不足 / 越界不堆积 / 不冻结时间轴
go test ./fusion -count=1 -v -run 'Calibrating|OutOfRange|DoesNotFreeze'
# 容量与处理预算边界
go test ./fusion -count=1 -v -run 'BatchLimit|Backlog|OpBudget'
# 预测/更新/快照并发交错（数据竞争检测，只跑并发用例，避免重型回放拖慢）
go test ./fusion -race -count=1 -run 'TestConcurrent'
# 30 分钟、各路时变漂移真正叠加的长跑（含开/关对齐各一个）
go test ./fusion -count=1 -v -run 'TestLongRunHalfHour'
# 全量（含 30 分钟长跑，约数分钟；建议不带 -race，回放量大时很慢）
go test ./... -count=1 -v

go run ./cmd/demo                 # 端到端回放演示
```
