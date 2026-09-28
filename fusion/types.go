// Package fusion 实现多传感器位姿融合与时间同步的状态估计引擎。
package fusion

import "time"

// SensorKind 标识观测来源。
type SensorKind int

const (
	SensorIMU SensorKind = iota
	SensorOdometry
	SensorAbsolutePose
)

func (k SensorKind) String() string {
	switch k {
	case SensorIMU:
		return "imu"
	case SensorOdometry:
		return "odometry"
	case SensorAbsolutePose:
		return "absolute_pose"
	}
	return "unknown"
}

// Vec3 三维向量。
type Vec3 struct{ X, Y, Z float64 }

// Quat 单位四元数，w 为标量部。
type Quat struct{ W, X, Y, Z float64 }

// Pose 位姿：位置 + 姿态。
type Pose struct {
	Position Vec3
	Rotation Quat
}

// Observation 一路传感器观测。
type Observation struct {
	Kind    SensorKind
	Stamp   time.Time // 传感器时间戳（已按时钟模型修正前）
	Seq     uint64    // 单调序号，用于确定性地区分重复时间戳
	Accel   Vec3      // IMU: 比力 m/s^2
	Gyro    Vec3      // IMU: 角速度 rad/s
	LinVel  Vec3      // Odometry: 线速度（机体系）
	AngVel  Vec3      // Odometry: 角速度
	Pose    Pose      // AbsolutePose: 绝对位姿观测
	CovDiag [9]float64
}

// RejectReason 观测被拒绝/降级的可区分原因。
type RejectReason int

const (
	ReasonNone RejectReason = iota
	ReasonAccepted
	ReasonOutlierGate      // 门限拒绝（马氏距离超限）
	ReasonStale            // 陈旧观测（落后于已融合时间）
	ReasonDuplicate        // 重复时间戳且序号不更大
	ReasonTimestampRegress // 时间戳倒退
	ReasonInsufficient     // 数据不足（如缺少初始化观测）
	ReasonTimeUnavailable  // 时间不可用（时钟漂移超界等）
	ReasonSensorDegraded   // 传感器降级，观测被降权或拒绝
	ReasonLimitExceeded    // 规模/耗时超限（统称；单批超限即此值）
	ReasonAlignCalibrating // 在线时间对齐样本不足，尚不可信，暂不采用
	ReasonAlignOutOfRange  // 时钟漂移超出可信范围，对齐不可用
	ReasonBacklogFull      // 待处理积压过多（重排缓冲将满），整批拒绝
	ReasonBudgetExceeded   // 单次处理预算不够，整批拒绝
)

// ReasonBatchTooLarge 是“单批规模超限”的语义化别名，
// 与 ReasonLimitExceeded 同值，保证既有判定契约不变的同时可区分三类边界。
const ReasonBatchTooLarge = ReasonLimitExceeded

func (r RejectReason) String() string {
	switch r {
	case ReasonAccepted:
		return "accepted"
	case ReasonOutlierGate:
		return "outlier_gate"
	case ReasonStale:
		return "stale"
	case ReasonDuplicate:
		return "duplicate"
	case ReasonTimestampRegress:
		return "timestamp_regress"
	case ReasonInsufficient:
		return "insufficient_data"
	case ReasonTimeUnavailable:
		return "time_unavailable"
	case ReasonSensorDegraded:
		return "sensor_degraded"
	case ReasonLimitExceeded:
		return "limit_exceeded"
	case ReasonAlignCalibrating:
		return "align_calibrating"
	case ReasonAlignOutOfRange:
		return "align_out_of_range"
	case ReasonBacklogFull:
		return "backlog_full"
	case ReasonBudgetExceeded:
		return "budget_exceeded"
	}
	return "none"
}

// Decision 一次观测处理的判定结果。
type Decision struct {
	Obs      Observation
	Accepted bool
	Reason   RejectReason
	Maha     float64 // 马氏距离（若计算）
	Detail   string
}

// Limits 规模与耗时上限。
type Limits struct {
	MaxPendingObs  int           // 待排序缓冲上限
	MaxBatchSize   int           // 单次摄入批大小上限
	MaxProcessTime time.Duration // 单次处理耗时上限
}

// Config 引擎配置。
type Config struct {
	Limits        Limits
	ProcessNoise  [9]float64    // 预测过程噪声对角
	GateChi2      float64       // 异常门限（马氏平方）
	StaleAfter    time.Duration // 观测落后于此时长视为陈旧
	DegradeAfter  time.Duration // 某路缺失超过此时长进入降级
	MaxClockDrift time.Duration // 允许的最大时钟漂移
	InitWindow    int           // 初始化所需的最少观测数
	// InitFallbackAfter 无绝对位姿时零位姿兜底初始化前的最短等待时长，
	// 以便各路在线时钟在校准期攒够样本；为 0 时取默认 3s。
	InitFallbackAfter time.Duration
	// AlignBootstrapGrace 时钟引导宽限：provisional 路观测落后状态在
	// (StaleAfter, AlignBootstrapGrace) 内时视为疑似初始时钟偏移，
	// 允许进入缓冲仅用于学习、不参与更新；为 0 时默认 1s。
	AlignBootstrapGrace time.Duration
	// DegradedNoiseInflate 是传感器降级恢复期间观测噪声的膨胀倍数。
	DegradedNoiseInflate float64
	// ReorderDelay 是乱序等待窗口：时间戳晚于（最新观测-窗口）的
	// 观测保留在缓冲中等待乱序伙伴；为 0 时每批摄入后立即全部消费。
	ReorderDelay time.Duration
	// AlignEnabled 开启在线时间对齐。默认开启；显式置 false 时引擎
	// 退化为“传感器时间戳即统一时间”的旧行为，结果与旧版本逐位一致。
	AlignEnabled *bool
	// AlignRequireCalibration 严格模式：上电即要求各路先校准，
	// 样本不足前观测以 align_calibrating 暂不采用。默认 false 表示
	// 恒等先验（上电即信任、同时在线辨识），与既有融合行为逐位兼容。
	AlignRequireCalibration bool
	// AlignMinSamples 一路时钟模型被采纳前至少需要的有效样本数。
	AlignMinSamples int
	// AlignConfirmSamples 偏移估计连续显著（超过 AlignActivateAt）
	// 达到该样本数后，才从恒等先验切换到估计模型。
	AlignConfirmSamples int
	// AlignActivateAt 偏移显著判定阈值；低于该阈值始终使用恒等映射，
	// 保证无漂移输入的处理路径与旧版本完全一致。
	AlignActivateAt time.Duration
	// AlignMaxOffset 允许的对齐偏移范围 |offset| 上限，超出即判漂移越界。
	AlignMaxOffset time.Duration
	// AlignMaxSkew 允许的时钟斜率偏差 |skew| 上限（无量纲，如 5000e-6=0.5%）。
	AlignMaxSkew float64
	// AlignForgetting 递归最小二乘遗忘因子（0,1]，1 表示不遗忘）。
	AlignForgetting float64
	// AlignInitStd 时钟偏移先验标准差（统一时间秒）。
	AlignInitStd time.Duration
	// AlignProcessStd 时钟偏移随机游走过程噪声（每秒标准差，秒/√s）。
	AlignProcessStd float64
	// AlignMeasStd 时钟样本测量噪声标准差（秒）。
	AlignMeasStd time.Duration
	// AlignMaxHeld 校准期暂存队列上限（每路），防止待处理数据无界增长。
	AlignMaxHeld int
	// AlignMinMotion 生成时钟样本所需的最小运动速率
	// （位置 0.05m/s、姿态 0.02rad/s 量级；由实现内部使用）。
	AlignMinLinVel float64
	AlignMinAngVel float64
}

// AlignStatus 一路传感器时钟对齐的当前状态（诊断用）。
type AlignStatus struct {
	Kind      SensorKind
	Phase     string // "reference" / "calibrating" / "provisional" / "active" / "out_of_range"
	Offset    time.Duration
	Skew      float64
	Samples   int
	Confirmed int
	Mapped    bool   // 是否真正使用估计模型换算（false=恒等映射）
	Trusted   bool   // 换算结果是否可参与更新
	Detail    string // 不可信时的可区分原因
}

// Stats 引擎运行统计（长跑容量/稳定性诊断用）。
type Stats struct {
	PeakPending      int
	Ingested         int // 进入重排缓冲的观测数
	Consumed         int // 出队进入状态推进的观测数
	RejectedByReason map[RejectReason]int64
}
