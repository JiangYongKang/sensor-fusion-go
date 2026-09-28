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
## 在线时间对齐与统一时间语义

- **统一时间轴**：以 IMU 为参考时钟（reference，恒等可信）。每路传感器在线
  估计仿射时钟模型 `统一时间 = 锚点 + 偏移 b + 斜率 s·(本地读数−锚点)`。
  所有顺序判定（重排缓冲有效键、陈旧/倒退、降级、预测 dt）都使用**对齐后的
  统一时间**；Stamp 仅用于去重身份与换算输入。全序仍为 (对齐键, Seq, Kind)。
- **乱序**：摄入分两阶段——先入重排缓冲，再按统一时间全序出队消费；
  ReorderDelay > 0 时晚于「最新键 − ReorderDelay」的观测留缓冲等待伙伴。
- **在线估计（漂移运行中变化也跟得上）**：位姿观测在正确状态时间做创新对
  时间的一阶加权投影 τ=(rᵀW ḣ)/(ḣᵀW ḣ)，给出「本地读数−物理时间」的带符号
  偏移证据；激活后由带遗忘因子的二维 RLS 跟踪偏移与斜率。阶段机：
  - provisional（默认恒等先验）：上电即信任、按无偏移参与并在线辨识；
    **确认激活前 RLS 参数保持精确 0/1、映射逐位恒等**；只有连续
    AlignConfirmSamples 个显著、同向且通过 SNR 检验的证据才激活 active，
    偶发噪声/模型运动误差不会误激活。
  - calibrating（AlignRequireCalibration=true，或初始化后才首次出现的路）：
    样本不足期间观测以 align_calibrating **暂不采用**，进有界校准队列，只
    学习时钟、不参与排序与状态更新。
- **重复/倒退/陈旧**：(Kind,Stamp,Seq) 相同以 duplicate 拒绝；换算键早于已
  出队序列以 timestamp_regress、落后超 StaleAfter 以 stale 拒绝。**模型更新后
  换算键落到状态过去的观测在消费阶段拒绝——状态只沿统一时间单调前进、绝不
  回退、已处理时间段绝不重复消费。**
- **漂移超界 vs 样本不足可区分**：偏移/斜率超 AlignMaxOffset/AlignMaxSkew →
  align_out_of_range（粘滞不恢复）；样本不足 → align_calibrating；换算后仍超
  硬漂移 MaxClockDrift → time_unavailable。
- **零漂移逐位兼容**：默认模式未激活的非参考路始终走精确恒等映射（整数时间
  算术，无浮点分支），融合姿态/速度/协方差与关闭对齐逐位一致。
- 相同输入重复运行结果严格一致（无随机性、无墙钟依赖；姿态与协方差逐位相等）。

## 容量、积压与处理预算

- **单批规模**：超 MaxBatchSize 整批 limit_exceeded。
- **处理预算**：按确定性成本模型（固定 obsProcessCost 微秒/条）估算，
  n·cost > MaxProcessTime 整批 budget_exceeded；不用墙钟，结果可复现，与
  规模超限是两个不同原因。
- **待处理积压**：摄入后待处理（重排缓冲 + 校准队列）将超 MaxPendingObs 时
  backlog_full；校准队列另有 AlignMaxHeld 上界，防止暂不采用观测无界增长。
- **整批原子性**：三类边界都在任何状态变更前判定，被拒批次不留部分更新；
  拒绝后引擎仍能处理后续小批次。Stats() 暴露待处理峰值、摄入/消费量与各原因
  计数，长跑可据此验证待处理不随时间增长。

## 异常判定与降级策略

- **门限剔除**：d²=rᵀS⁻¹r 超 GateChi2（默认 16，约 6 维 3σ）以 outlier_gate
  拒绝，状态与协方差不被污染（判定在预测之后、误差注入之前）。
- **Joseph 形式协方差**：P ← (I−KH)P(I−KH)ᵀ + KRKᵀ，对称化并钳制对角下界。
- **缺测降级**：某路间隔超 DegradeAfter 即降级，观测噪声膨胀
  DegradedNoiseInflate 倍后参与（sensor_degraded），避免陈旧数据主导又允许恢复。
- **初始化**：收满 InitWindow 且含已对齐可信绝对位姿时初始化；无位姿按观测
  数（及可选 InitFallbackAfter）零位姿兜底。
- 判定原因枚举：accepted / outlier_gate / stale / duplicate /
  timestamp_regress / insufficient_data / time_unavailable / sensor_degraded /
  align_calibrating / align_out_of_range / limit_exceeded / backlog_full /
  budget_exceeded，均可通过 Decisions() 区分。

## 并发一致性

- 全部状态变更在写锁内原子完成，快照读取走读锁；三类边界校验也在锁内、任何
  状态变更之前，跨 Ingest/AdvanceTo/Snapshot/Stats/AlignStatuses 交错无半更新。
  Snapshot() 返回深拷贝。-race 覆盖多写入方 + 并发推进 + 高并发快照/诊断交错。

## 回放数据与验证方法

- GenerateReplay(ReplayConfig) 确定性生成多路回放：真值轨迹为水平圆周 + 垂直
  正弦（可解析求导），采样 IMU（比力=Rᵀ(a−g)）、机体系里程计、世界系绝对位姿；
  叠加高斯噪声、丢帧（DropProb）、异常值（OutlierProb/OutlierScale）、局部乱序
  （ReorderWindow）、**每路独立且随时间变化方向可不一致的时钟漂移**（Clocks：
  分段 ClockKnot{At,Offset,Skew}，兼容旧线性 ClockDrift）、某路长时间缺测
  （PoseGap/PoseGapAt、OdomGap/OdomGapAt）。相同 Seed 生成相同序列。
- 测试覆盖（fusion/ 包内）：
  - 四元数/矩阵数值稳定性；乱序殊途同归与 ReorderDelay 窗口；
  - 重复/倒退/陈旧/漂移可区分拒绝；异常门限拒绝不污染状态；
  - **在线对齐**：时钟模型（固定偏移、时变斜率反转、越界粘滞、零漂移恒等）、
    引擎级时变漂移精度、样本不足 vs 越界区分、模型切换后时间单调不回退、
    无漂移逐位一致、重复运行姿态/协方差严格一致；
  - 规模/积压/预算三类边界整批拒绝、无部分更新、拒绝后可继续；
  - 预测/更新/快照/诊断交错并发（-race）；缺测降级恢复；
  - 长跑：TestLongRunFusion（60s，位置<2m、姿态<0.2rad）与
    TestLongRun30Min（**30 分钟、约 22 万观测**，待处理峰值有界、四元数单位
    范数、速度/位置/协方差稳定、误差有界、重复运行严格一致）。
- 运行验证：

```sh
go test ./...                     # 全部测试（日志含关键输入与判定依据）
go test ./... -race -count=1      # 全部测试 + 竞态检测
go test ./... -short              # 跳过 30 分钟长跑
go test . -run TestLongRun30Min   # 30 分钟 / 约 22 万观测长跑
go run ./cmd/demo                 # 端到端回放（含在线对齐状态打印）
```
