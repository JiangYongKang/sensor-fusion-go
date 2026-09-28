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
  传感器在线估计其时钟到统一时间轴的关系，换算后的时间真正、且唯一地
  参与状态推进、顺序判定、陈旧/倒退判定与降级判定——**对外状态显示用
  什么时间，融合内部就真的用什么时间**。
  - 以一路（默认 IMU，可用 `AlignAutoRef/AlignRef` 选择）为时间基准，
    其余为待对齐路。
  - 利用各路共享的机身偏航运动做跨路时延估计：信号按时间有序缓存，
    连续偏航角由有序环统一积分/解卷绕（免疫批内乱序），在居中窗口上
    做去均值形状相关 + 联合 (skew, lag) 网格搜索得到锚点；锚点一旦采纳
    即不可变，换算以“锚点之间的分段插值”为准（局部段斜率能即时跟上
    漂移换向），全局稳健斜率只用于越界诊断与跨度外兜底。
  - **漂移检测严格、零漂移零影响**：只有当近期锚点实测偏移（lag 的同号
    中位数，直接就是 unified−raw 当前偏移，与拟合斜率/原点无关）显著
    超过 `AlignDetect` 且方向同号一致时，才认定该路存在漂移。零漂移下
    相关 lag 是符号反复翻转的零均值噪声，永不触发，因此该路始终恒等、
    零额外开销，**开关对齐的融合结果逐位一致**（有自动化用例严格断言）。
  - **状态机（原因可区分，绝不把未对齐数据当已对齐）**：
    `cold`（无漂移证据，恒等放行）→
    `calibrating/align_calibrating`（已发现漂移但样本不足，观测**暂缓、
    不参与任何更新**，与“超界”明确区分）→ `locked`（偏移、残差、斜率/
    偏移范围均达标，按换算时间消费）；斜率或偏移超
    `AlignMaxSkew/AlignMaxDrift` 判 `out_of_range / time_unavailable`。
  - **换算时间绝不回跳（时变漂移稳定的关键）**：在线只消费“落在已采纳
    锚点覆盖区间（最新锚点 + 半窗短距外推）内”的观测——锚点间插值由
    不可变锚点唯一确定，换算时间随后续数据不会改变；更晚的点先要靠右外
    推、其结果会随新锚点变化，在线先暂缓，避免高频速度更新用会回跳的时
    间把“未来速度”配到“过去姿态”，经 IMU 积分正反馈放大而在漂移换向点
    发散。离线追跑由 `AdvanceTo` 切到“排空口径”一次性放出暂缓尾部。
  - **校准宽限**：某路首次出现后 `AlignCalibrateAfter`（默认 5s）内
    暂不采用；超过宽限仍未锁定，则按原始时间（恒等）**降级放行**，以可
    区分原因 `align_degraded` 记录并膨胀噪声，保证对齐层永不饿死滤波
    器（只有真正越界才持续拒绝）。
  - 一路长时间无数据时，对齐水线在 `DegradeAfter+AlignSearchRange`
    后不再被该路冻结，避免单路缺测阻塞其他路、撑爆缓冲；在线锚点确认的
    暂态延迟约半个相关窗（实时 1× 速率下待处理量有界），不会卡住整条
    时间轴，也不会让待处理数据无限堆积。
  - 相同输入重复运行严格一致（对齐相关为纯确定性计算，不读墙钟、无
    随机数）。
  - 配置：`AlignDetect / AlignMinSamples / AlignMinSpan / AlignTolerance
    / AlignMaxSkew / AlignMaxDrift / AlignWindow / AlignSearchRange /
    AlignReleaseMargin / AlignCalibrateAfter / AlignAutoRef / AlignRef`。
  - 查询：`Engine.AlignmentStats()` 返回每路状态/skew/offset/残差/样本；
    `Engine.AlignedTime(kind, raw)` 在该路 locked/cold 时返回换算时间。
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
  insufficient_data / align_calibrating / align_degraded / time_unavailable /
  sensor_degraded / limit_exceeded / backlog_exceeded / process_budget`。
  其中三类时间对齐口径明确区分：“未发现漂移、恒等采用”（cold，按
  accepted）、“已发现漂移但样本不足暂不采用”(`align_calibrating`)、
  “校准超时未锁定按原始时间降级放行”(`align_degraded`)、“漂移超界”
  (`time_unavailable`)，绝不在状态不可信时对外显示“已锁定”。

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
  - **修复回归（直接断言用户反馈的三类问题）**：
    **无漂移输入开/关对齐逐位一致且各路保持 cold、绝不误锁**
    （`TestAlignmentNoDriftBitIdentical`，多 seed × 多时长）；
    **漂移随时间换向时开对齐的融合位置误差不劣于关对齐、速度有界不
    发散、且对外确为 locked**（`TestAlignmentTimeVaryingNotWorse`，
    1 分钟与 3 分钟）；
    **半小时回放把各路时变、换向、偏移有界的时钟漂移真实叠入，开对齐
    全程有界、末端优于关对齐、待处理有上限、重复运行一致**
    （`TestAlignmentHalfHourReversalBounded`，>22 万观测）；
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
