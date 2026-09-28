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

// AlignRefAuto 表示在线对齐参考路自动选择：优先 IMU，无 IMU 时
// 取最先到达的一路。配置 AlignRef 为该值（默认）即启用此策略。
const AlignRefAuto SensorKind = -1

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
	ReasonLimitExceeded    // 单批规模超限
	ReasonAlignCalibrating // 在线时间对齐样本不足，暂不采用
	ReasonBacklogExceeded  // 待处理积压过多，整批拒绝
	ReasonProcessBudget    // 处理预算不足，整批拒绝
)

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
	case ReasonBacklogExceeded:
		return "backlog_exceeded"
	case ReasonProcessBudget:
		return "process_budget"
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
	Aligned  time.Time // 在线时间对齐换算后的统一时间（未换算时为零值）
	RawStamp time.Time // 观测原始传感器时间戳
}

// AlignState 一路传感器相对统一时间轴的在线对齐状态。
type AlignState int

const (
	AlignCold        AlignState = iota // 未发现漂移，按恒等映射处理（兼容模式）
	AlignCalibrating                   // 已发现偏差，样本不足/残差过大，观测暂不采用
	AlignLocked                        // 映射可信，观测按换算时间参与更新
	AlignOutOfRange                    // 映射参数超出可信范围，观测暂不采用
)

func (s AlignState) String() string {
	switch s {
	case AlignCold:
		return "cold"
	case AlignCalibrating:
		return "calibrating"
	case AlignLocked:
		return "locked"
	case AlignOutOfRange:
		return "out_of_range"
	}
	return "unknown"
}

// AlignStat 一路传感器的在线对齐状态快照。
// 统一时间 = 原始时间戳 × Skew + Offset。
type AlignStat struct {
	Kind     SensorKind
	State    AlignState
	Skew     float64       // 时钟斜率估计（1 表示无快慢）
	Offset   time.Duration // 时钟截距（统一时间 − 原始时间戳在零点处的差）
	Residual time.Duration // 最近窗口拟合残差（RMS）
	Samples  int           // 有效配对样本数
	IsRef    bool          // 是否为统一时间轴参考路
}

// EngineStats 引擎运行期规模统计（确定性，仅随输入变化）。
type EngineStats struct {
	Pending       int // 当前待处理（未消费）观测数
	PendingHigh   int // 运行以来待处理观测峰值
	Consumed      int // 已消费观测总数
	Rejected      int // 已拒绝观测总数（含降级拒绝）
	MaxBatchSize  int
	MaxPendingObs int
}

// Limits 规模与耗时上限。
type Limits struct {
	MaxPendingObs  int           // 待排序缓冲上限
	MaxBatchSize   int           // 单次摄入批大小上限
	MaxProcessTime time.Duration // 单次处理墙钟预算上限（0=用默认）
	// MaxBatchOps 单次处理的确定性操作预算（对齐时延搜索/更新等重操作
	// 的折算次数）。>0 时优先于墙钟判定，使预算超限在相同输入下可复现。
	MaxBatchOps int
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
	// DegradedNoiseInflate 是传感器降级恢复期间观测噪声的膨胀倍数。
	DegradedNoiseInflate float64
	// ReorderDelay 是乱序等待窗口：时间戳晚于（最新观测-窗口）的
	// 观测保留在缓冲中等待乱序伙伴；为 0 时每批摄入后立即全部消费。
	ReorderDelay time.Duration
	// EnableAlignment 启用在线时间对齐层。关闭时全部观测按原始
	// 时间戳处理，行为与旧版完全一致（兼容基线）。
	EnableAlignment bool
	// AlignMinSamples 单路映射可信所需的最少配对样本数。
	AlignMinSamples int
	// AlignMinSpan 可信映射所需的最小样本时间跨度。
	AlignMinSpan time.Duration
	// AlignDetect 统一时间轴偏差超过该阈值时开始在线校准；
	// 偏差不超过阈值的路保持恒等映射（零漂移输入行为不变）。
	AlignDetect time.Duration
	// AlignTolerance 映射拟合残差可信门限。
	AlignTolerance time.Duration
	// AlignMaxSkew 可信斜率范围（相对 1 的最大偏差，如 0.05 = ±5%）。
	AlignMaxSkew float64
	// AlignWindow 跟踪漂移变化的样本时间窗（窗内最小二乘）。
	AlignWindow time.Duration
	// AlignReleaseMargin 按各路原始观测前沿放水的安全余量。
	AlignReleaseMargin time.Duration
	// AlignMaxDrift 可信时钟映射的最大偏移量，超出为 out_of_range。
	AlignMaxDrift time.Duration
	// AlignSearchRange 时延相关搜索半径：只需覆盖运行中可能遇到的
	// 时钟偏差量级，取明显小于信号特征周期即可避开周期假峰。
	AlignSearchRange time.Duration
	// AlignCalibrateAfter 对齐校准宽限：某路首次出现后这段时间内若未
	// 锁定，观测暂缓（align_calibrating）；超过宽限仍不可信，则不再
	// 饿死融合，按原始时间（恒等）降级放行，只有漂移越界才持续拒绝。
	AlignCalibrateAfter time.Duration
	// AlignRef 指定统一时间轴参考路；AlignAutoRef 为 true 时忽略本
	// 字段，自动选择（IMU 优先，无 IMU 取最先到达的一路）。参考路
	// 恒为恒等映射。
	AlignRef     SensorKind
	AlignAutoRef bool
}
