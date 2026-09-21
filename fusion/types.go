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
	ReasonLimitExceeded    // 规模/耗时超限
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
	// DegradedNoiseInflate 是传感器降级恢复期间观测噪声的膨胀倍数。
	DegradedNoiseInflate float64
	// ReorderDelay 是乱序等待窗口：时间戳晚于（最新观测-窗口）的
	// 观测保留在缓冲中等待乱序伙伴；为 0 时每批摄入后立即全部消费。
	ReorderDelay time.Duration
}
