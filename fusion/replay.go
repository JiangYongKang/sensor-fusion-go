package fusion

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

// ClockKnot 描述某路传感器时钟在一个时刻的状态：
// 在 Knot 时刻传感器读数相对真值的偏移 Offset；随后该路时钟以
// Skew（相对真值的速率，0=不漂移）运行，直到下一个节点。
// 多个节点即可模拟“快慢在运行中变化、方向/幅度不一致”的设备时钟。
type ClockKnot struct {
	At     time.Duration // 相对回放起点
	Offset time.Duration // 该时刻读数偏移
	Skew   float64       // 下一区间的速率偏差（如 +500e-6 表示快 500ppm）
}

// ClockProfile 一路时钟的分段线性漂移曲线；为空表示无漂移（逐位恒等）。
type ClockProfile []ClockKnot

// offsetAt 计算真值时刻 t（相对起点）对应的传感器读数偏移。
func (p ClockProfile) offsetAt(t time.Duration) time.Duration {
	if len(p) == 0 {
		return 0
	}
	if t <= p[0].At {
		return p[0].Offset
	}
	for i := 0; i < len(p)-1; i++ {
		if t >= p[i].At && t < p[i+1].At {
			dt := t - p[i].At
			return p[i].Offset + time.Duration(p[i].Skew*float64(dt))
		}
	}
	last := p[len(p)-1]
	return last.Offset + time.Duration(last.Skew*float64(t-last.At))
}

// ReplayConfig 本地传感器回放数据生成配置。
// 相同配置（含 Seed）必然生成完全相同的观测序列，保证可复现。
type ReplayConfig struct {
	Seed          int64
	Duration      time.Duration
	IMURate       float64 // Hz
	OdomRate      float64
	PoseRate      float64
	DropProb      float64       // 丢帧概率（里程计/绝对位姿）
	OutlierProb   float64       // 异常值概率
	OutlierScale  float64       // 异常幅度（米 / 弧度量级）
	ReorderWindow int           // 乱序窗口（条）：局部洗牌幅度
	ClockDrift    time.Duration // 兼容字段：全程线性累计偏移（仅绝对位姿路）
	PoseGap       time.Duration // 绝对位姿中途缺测时长（模拟传感器降级）
	// OdomGap 里程计中途缺测时长（模拟某路长时间无数据）。
	OdomGap time.Duration
	// PoseGapAt / OdomGapAt 指定缺测窗口起点；为 0 时默认在回放中点。
	PoseGapAt time.Duration
	OdomGapAt time.Duration
	// Clocks 各路随时间变化的漂移曲线；设置后优先于 ClockDrift。
	Clocks map[SensorKind]ClockProfile
}

// Trajectory 真值轨迹：绕原点水平圆周 + 垂直正弦起伏，可解析求导。
type Trajectory struct {
	Start    time.Time
	Radius   float64 // 圆周半径 m
	Omega    float64 // 角速度 rad/s
	AmpZ     float64 // 垂直振幅 m
	FreqZ    float64 // 垂直频率 rad/s
	Duration time.Duration
}

// DefaultTrajectory 默认真值轨迹。
func DefaultTrajectory(start time.Time, dur time.Duration) Trajectory {
	return Trajectory{
		Start:    start,
		Radius:   5.0,
		Omega:    0.3,
		AmpZ:     0.5,
		FreqZ:    0.5,
		Duration: dur,
	}
}

// TruePoseAt 返回轨迹在时刻 t 的真值位姿（朝向沿切线方向）。
func (tr Trajectory) TruePoseAt(t time.Time) Pose {
	tau := t.Sub(tr.Start).Seconds()
	x := tr.Radius * math.Cos(tr.Omega*tau)
	y := tr.Radius * math.Sin(tr.Omega*tau)
	z := tr.AmpZ * math.Sin(tr.FreqZ*tau)
	yaw := tr.Omega*tau + math.Pi/2 // 切线方向
	return Pose{
		Position: Vec3{x, y, z},
		Rotation: QuatFromRotVec(Vec3{Z: yaw}),
	}
}

// TrueVelAt 真值速度（世界系，解析导数）。
func (tr Trajectory) TrueVelAt(t time.Time) Vec3 {
	tau := t.Sub(tr.Start).Seconds()
	return Vec3{
		X: -tr.Radius * tr.Omega * math.Sin(tr.Omega*tau),
		Y: tr.Radius * tr.Omega * math.Cos(tr.Omega*tau),
		Z: tr.AmpZ * tr.FreqZ * math.Cos(tr.FreqZ*tau),
	}
}

// TrueAccelAt 真值加速度（世界系，解析导数）。
func (tr Trajectory) TrueAccelAt(t time.Time) Vec3 {
	tau := t.Sub(tr.Start).Seconds()
	w2 := tr.Omega * tr.Omega
	return Vec3{
		X: -tr.Radius * w2 * math.Cos(tr.Omega*tau),
		Y: -tr.Radius * w2 * math.Sin(tr.Omega*tau),
		Z: -tr.AmpZ * tr.FreqZ * tr.FreqZ * math.Sin(tr.FreqZ*tau),
	}
}

// Generate 观测噪声标准差（各路）。
const (
	replayIMUAccelNoise = 0.05
	replayIMUGyroNoise  = 0.005
	replayOdomNoise     = 0.05
	replayPosePosNoise  = 0.10
	replayPoseRotNoise  = 0.02
)

// GenerateReplay 生成确定性的多路乱序回放数据。
// 生成流程：真值采样 → 加噪 → 丢帧 → 注入异常 → 时钟漂移 → 局部乱序。
// 每一步都仅依赖 cfg 与固定算法，与运行环境无关。
func GenerateReplay(cfg ReplayConfig) []Observation {
	rng := rand.New(rand.NewSource(cfg.Seed))
	start := time.Unix(1_700_000_000, 0)
	traj := DefaultTrajectory(start, cfg.Duration)
	clocks := cfg.Clocks
	if clocks == nil {
		clocks = map[SensorKind]ClockProfile{}
	}
	// stamped 按该路时钟曲线把真值时刻换算为传感器读数；
	// 物理量仍在真值时刻采样（时间戳与物理量解耦，模拟真实设备）。
	stamped := func(k SensorKind, t time.Time) time.Time {
		if p, ok := clocks[k]; ok && len(p) > 0 {
			return t.Add(p.offsetAt(t.Sub(start)))
		}
		if k == SensorAbsolutePose && cfg.ClockDrift != 0 {
			frac := t.Sub(start).Seconds() / cfg.Duration.Seconds()
			return t.Add(time.Duration(frac * float64(cfg.ClockDrift)))
		}
		return t
	}

	var out []Observation
	var seq uint64
	next := func() uint64 { s := seq; seq++; return s }

	// --- IMU：比力 = Rᵀ(a_world − g)，角速度 = 偏航角速度（机体系 z） ---
	if cfg.IMURate > 0 {
		dt := time.Duration(float64(time.Second) / cfg.IMURate)
		for t := start; t.Before(start.Add(cfg.Duration)); t = t.Add(dt) {
			pose := traj.TruePoseAt(t)
			accW := traj.TrueAccelAt(t)
			gravity := Vec3{Z: -9.81}
			specific := QuatRotate(QuatConj(pose.Rotation), vecSub(accW, gravity))
			gyro := Vec3{Z: traj.Omega}
			out = append(out, Observation{
				Kind:  SensorIMU,
				Stamp: stamped(SensorIMU, t),
				Seq:   next(),
				Accel: Vec3{
					specific.X + rng.NormFloat64()*replayIMUAccelNoise,
					specific.Y + rng.NormFloat64()*replayIMUAccelNoise,
					specific.Z + rng.NormFloat64()*replayIMUAccelNoise,
				},
				Gyro: Vec3{
					gyro.X + rng.NormFloat64()*replayIMUGyroNoise,
					gyro.Y + rng.NormFloat64()*replayIMUGyroNoise,
					gyro.Z + rng.NormFloat64()*replayIMUGyroNoise,
				},
			})
		}
	}

	// --- 里程计：机体系线速度 ---
	if cfg.OdomRate > 0 {
		dt := time.Duration(float64(time.Second) / cfg.OdomRate)
		ogAt := cfg.OdomGapAt
		if ogAt == 0 {
			ogAt = cfg.Duration / 2
		}
		ogStart := start.Add(ogAt)
		ogEnd := ogStart.Add(cfg.OdomGap)
		for t := start; t.Before(start.Add(cfg.Duration)); t = t.Add(dt) {
			if cfg.OdomGap > 0 && !t.Before(ogStart) && t.Before(ogEnd) {
				continue
			}
			if rng.Float64() < cfg.DropProb {
				continue // 丢帧
			}
			pose := traj.TruePoseAt(t)
			velB := QuatRotate(QuatConj(pose.Rotation), traj.TrueVelAt(t))
			o := Observation{
				Kind:  SensorOdometry,
				Stamp: stamped(SensorOdometry, t),
				Seq:   next(),
				LinVel: Vec3{
					velB.X + rng.NormFloat64()*replayOdomNoise,
					velB.Y + rng.NormFloat64()*replayOdomNoise,
					velB.Z + rng.NormFloat64()*replayOdomNoise,
				},
				CovDiag: [9]float64{replayOdomNoise * replayOdomNoise,
					replayOdomNoise * replayOdomNoise, replayOdomNoise * replayOdomNoise},
			}
			if rng.Float64() < cfg.OutlierProb {
				o.LinVel.X += cfg.OutlierScale * (2*rng.Float64() - 1) // 异常值
			}
			out = append(out, o)
		}
	}

	// --- 绝对位姿：含中途缺测（降级模拟）与时钟漂移 ---
	if cfg.PoseRate > 0 {
		dt := time.Duration(float64(time.Second) / cfg.PoseRate)
		gAt := cfg.PoseGapAt
		if gAt == 0 {
			gAt = cfg.Duration / 2
		}
		gapStart := start.Add(gAt)
		gapEnd := gapStart.Add(cfg.PoseGap)
		for t := start; t.Before(start.Add(cfg.Duration)); t = t.Add(dt) {
			if cfg.PoseGap > 0 && !t.Before(gapStart) && t.Before(gapEnd) {
				continue // 缺测窗口
			}
			if rng.Float64() < cfg.DropProb {
				continue
			}
			pose := traj.TruePoseAt(t)
			o := Observation{
				Kind:  SensorAbsolutePose,
				Stamp: stamped(SensorAbsolutePose, t),
				Seq:   next(),
				Pose: Pose{
					Position: Vec3{
						pose.Position.X + rng.NormFloat64()*replayPosePosNoise,
						pose.Position.Y + rng.NormFloat64()*replayPosePosNoise,
						pose.Position.Z + rng.NormFloat64()*replayPosePosNoise,
					},
					Rotation: QuatMul(pose.Rotation, QuatFromRotVec(Vec3{
						X: rng.NormFloat64() * replayPoseRotNoise,
						Y: rng.NormFloat64() * replayPoseRotNoise,
						Z: rng.NormFloat64() * replayPoseRotNoise,
					})),
				},
				CovDiag: [9]float64{
					replayPosePosNoise * replayPosePosNoise,
					replayPosePosNoise * replayPosePosNoise,
					replayPosePosNoise * replayPosePosNoise,
					replayPoseRotNoise * replayPoseRotNoise,
					replayPoseRotNoise * replayPoseRotNoise,
					replayPoseRotNoise * replayPoseRotNoise,
				},
			}
			if rng.Float64() < cfg.OutlierProb {
				o.Pose.Position.X += cfg.OutlierScale * (2*rng.Float64() - 1)
				o.Pose.Position.Y += cfg.OutlierScale * (2*rng.Float64() - 1)
			}
			out = append(out, o)
		}
	}

	// --- 按 (时间戳, 序号) 归并三路数据流，再施加局部乱序 ---
	sort.SliceStable(out, func(i, j int) bool {
		if !out[i].Stamp.Equal(out[j].Stamp) {
			return out[i].Stamp.Before(out[j].Stamp)
		}
		return out[i].Seq < out[j].Seq
	})

	// --- 局部乱序：滑动窗口内确定性洗牌 ---
	if cfg.ReorderWindow > 1 {
		for i := 0; i < len(out); i += cfg.ReorderWindow {
			end := i + cfg.ReorderWindow
			if end > len(out) {
				end = len(out)
			}
			// 使用独立 rng 状态推进，保证窗口划分稳定
			rng.Shuffle(end-i, func(a, b int) {
				out[i+a], out[i+b] = out[i+b], out[i+a]
			})
		}
	}
	return out
}
