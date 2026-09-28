package fusion

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

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
	ClockDrift    time.Duration // 全程累计时钟漂移（线性，作用于绝对位姿路；被 Clocks 覆盖）
	PoseGap       time.Duration // 绝对位姿中途缺测时长（模拟传感器降级）
	// OdomAngVel 为 true 时里程计观测带角速度（用于丰富共享运动信号）。
	OdomAngVel bool
	// SensorGap 额外模拟某路中途长时间无数据（除 PoseGap 外）。
	SensorGap map[SensorKind]time.Duration
	// Clocks 各路独立时钟漂移曲线；未配置的路时钟与统一时间一致。
	Clocks map[SensorKind]ClockProfile
	// Excited 使用时变激励轨迹（角速度随时间变化），供在线时间
	// 对齐的跨路运动信号相关使用；false 时保持旧的等速圆周轨迹。
	Excited bool
}

// Clock 返回某路的时钟漂移曲线。绝对位姿路在未配置 Clocks 时
// 兼容旧版线性 ClockDrift（斜率恒定）。
func (cfg ReplayConfig) Clock(k SensorKind) (ClockProfile, bool) {
	if cp, ok := cfg.Clocks[k]; ok {
		return cp, true
	}
	if k == SensorAbsolutePose && cfg.ClockDrift != 0 {
		return ClockProfile{
			Slopes:   []float64{1 + cfg.ClockDrift.Seconds()/cfg.Duration.Seconds()},
			Segments: nil,
		}, true
	}
	return ClockProfile{}, false
}

// ClockProfile 一路传感器相对统一时间轴的时钟漂移曲线。
// 统一时间 = skew(物理时刻) × 原始时间戳 + offset；回放按分段常数
// 斜率生成原始时间戳（时钟在运行中忽快忽慢、可换向），斜率 1 表示
// 无快慢，Offset0 为初始固定偏差。不同路使用不同曲线。
type ClockProfile struct {
	Slopes   []float64     // 各段时钟斜率（如 1.002 表示每秒快 2ms）
	Segments []Duration    // 各段物理时长（长度应与 Slopes 相同或短一）
	Offset0  time.Duration // 初始固定偏差：统一时间 − 原始时间戳
}

// Duration 是 time.Duration 的别名，仅为配置字面量更短。
type Duration = time.Duration

// offsetAt 物理时刻 t 相对 start 的漂移量（统一时间 − 原始时间戳）。
// offset(t) = ∫₀ᵗ(1 − s(u))du，斜率 Slopes[i] 在时间段 Segments[i]
// 内生效，末段斜率延伸到全程结束，分段点处偏移连续。
func (c ClockProfile) offsetAt(start, t time.Time) time.Duration {
	tau := t.Sub(start).Seconds()
	if len(c.Slopes) == 0 {
		return c.Offset0
	}
	var offset, segStart float64
	for i, Ld := range c.Segments {
		L := Ld.Seconds()
		if tau < segStart+L {
			offset += (1 - c.Slopes[i]) * (tau - segStart)
			return c.Offset0 + time.Duration(offset*float64(time.Second))
		}
		offset += (1 - c.Slopes[i]) * L
		segStart += L
	}
	// 已过所有分段：最后一个斜率继续生效。
	s := c.Slopes[len(c.Slopes)-1]
	offset += (1 - s) * (tau - segStart)
	return c.Offset0 + time.Duration(offset*float64(time.Second))
}

// rawStamp 返回物理时刻 t 在该路时钟上的原始时间戳。
// 原始时间戳 = 物理时刻 − (统一时间 − 原始时间戳)。
func (c ClockProfile) rawStamp(start, t time.Time) time.Time {
	return t.Add(-c.offsetAt(start, t))
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

// excitedTrajectory 时变激励轨迹：偏航速率随时间变化
// ω(τ)=0.3+0.25·sin(0.13τ)，使角速度信号可用于跨路时延相关。
type excitedTrajectory struct{ start time.Time }

// ExcitedTrajectory 构造时变激励轨迹（角速度/加速度均随时间变化）。
func ExcitedTrajectory(start time.Time) excitedTrajectory { return excitedTrajectory{start: start} }

func (tr excitedTrajectory) tau(t time.Time) float64 { return t.Sub(tr.start).Seconds() }

// yawRate 时变偏航角速度（频率更高，使短时信号窗形状可辨识）。
func (tr excitedTrajectory) yawRate(tau float64) float64 { return 0.3 + 0.22*math.Sin(0.6*tau) }

// yawRateDot 偏航角加速度。
func (tr excitedTrajectory) yawRateDot(tau float64) float64 { return 0.22 * 0.6 * math.Cos(0.6*tau) }

// yaw 解析积分的偏航角：∫ω = 0.3τ + (0.22/0.6)(1−cos(0.6τ))。
func (tr excitedTrajectory) yaw(tau float64) float64 {
	return 0.3*tau + (0.22/0.6)*(1-math.Cos(0.6*tau))
}

// PoseAt 时变轨迹在物理时刻 t 的位姿（半径圆周 + 垂直正弦）。
func (tr excitedTrajectory) PoseAt(t time.Time) Pose {
	tau := tr.tau(t)
	x := 5.0 * math.Cos(tr.yaw(tau))
	y := 5.0 * math.Sin(tr.yaw(tau))
	z := 0.5 * math.Sin(0.5*tau)
	return Pose{Position: Vec3{x, y, z}, Rotation: QuatFromRotVec(Vec3{Z: tr.yaw(tau)})}
}

// VelAt 世界系线速度：沿切向速度 R·ω 加垂直分量。
func (tr excitedTrajectory) VelAt(t time.Time) Vec3 {
	tau := tr.tau(t)
	w := tr.yawRate(tau)
	return Vec3{
		X: -5.0 * w * math.Sin(tr.yaw(tau)),
		Y: 5.0 * w * math.Cos(tr.yaw(tau)),
		Z: 0.5 * 0.5 * math.Cos(0.5*tau),
	}
}

// YawRate 返回机体系 z 角速度。
func (tr excitedTrajectory) YawRate(t time.Time) float64 { return tr.yawRate(tr.tau(t)) }

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
	replayPoseRotNoise  = 0.006
)

// GenerateReplay 生成确定性的多路乱序回放数据。
// 生成流程：真值采样 → 加噪 → 丢帧 → 注入异常 → 各路时钟漂移 → 局部乱序。
// 每一步都仅依赖 cfg 与固定算法，与运行环境无关。
func GenerateReplay(cfg ReplayConfig) []Observation {
	rng := rand.New(rand.NewSource(cfg.Seed))
	start := time.Unix(1_700_000_000, 0)
	traj := DefaultTrajectory(start, cfg.Duration)

	// rawFor 给出某路在物理时刻 t 上的原始时间戳；未配置漂移曲线
	// 的路（IMU/里程计默认，位姿路兼容旧 ClockDrift）保持物理时间。
	rawFor := func(k SensorKind, t time.Time) time.Time {
		if cp, ok := cfg.Clock(k); ok {
			return cp.rawStamp(start, t)
		}
		return t
	}
	gapped := func(k SensorKind, t time.Time) bool {
		if g, ok := cfg.SensorGap[k]; ok && g > 0 {
			gs := start.Add(cfg.Duration / 2)
			if !t.Before(gs) && t.Before(gs.Add(g)) {
				return true
			}
		}
		return false
	}
	ex := ExcitedTrajectory(start)
	// poseAt/velAt/accAt/yawRateAt 按是否时变激励取对应真值。
	poseAt := func(t time.Time) Pose {
		if cfg.Excited {
			return ex.PoseAt(t)
		}
		return traj.TruePoseAt(t)
	}
	velAt := func(t time.Time) Vec3 {
		if cfg.Excited {
			return ex.VelAt(t)
		}
		return traj.TrueVelAt(t)
	}
	yawRateAt := func(t time.Time) float64 {
		if cfg.Excited {
			return ex.YawRate(t)
		}
		return traj.Omega
	}
	// accAt：默认轨迹用解析加速度；激励轨迹用世界系速度的中心差分，
	// 仅用于 IMU 比力回放（差分误差远小于设定的加噪水平）。
	accAt := func(t time.Time) Vec3 {
		if !cfg.Excited {
			return traj.TrueAccelAt(t)
		}
		const h = 5 * time.Millisecond
		vp := ex.VelAt(t.Add(h))
		vm := ex.VelAt(t.Add(-h))
		dt := 2 * h.Seconds()
		return Vec3{X: (vp.X - vm.X) / dt, Y: (vp.Y - vm.Y) / dt, Z: (vp.Z - vm.Z) / dt}
	}

	var out []Observation
	var seq uint64
	next := func() uint64 { s := seq; seq++; return s }

	// --- IMU：比力 = Rᵀ(a_world − g)，角速度 = 偏航角速度（机体系 z） ---
	if cfg.IMURate > 0 {
		dt := time.Duration(float64(time.Second) / cfg.IMURate)
		for t := start; t.Before(start.Add(cfg.Duration)); t = t.Add(dt) {
			if gapped(SensorIMU, t) {
				continue
			}
			pose := poseAt(t)
			accW := accAt(t)
			gravity := Vec3{Z: -9.81}
			specific := QuatRotate(QuatConj(pose.Rotation), vecSub(accW, gravity))
			gyro := Vec3{Z: yawRateAt(t)}
			out = append(out, Observation{
				Kind:  SensorIMU,
				Stamp: rawFor(SensorIMU, t),
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
		for t := start; t.Before(start.Add(cfg.Duration)); t = t.Add(dt) {
			if gapped(SensorOdometry, t) {
				continue
			}
			if rng.Float64() < cfg.DropProb {
				continue // 丢帧
			}
			pose := poseAt(t)
			velB := QuatRotate(QuatConj(pose.Rotation), velAt(t))
			w := yawRateAt(t)
			o := Observation{
				Kind:  SensorOdometry,
				Stamp: rawFor(SensorOdometry, t),
				Seq:   next(),
				LinVel: Vec3{
					velB.X + rng.NormFloat64()*replayOdomNoise,
					velB.Y + rng.NormFloat64()*replayOdomNoise,
					velB.Z + rng.NormFloat64()*replayOdomNoise,
				},
				AngVel: Vec3{
					rng.NormFloat64() * replayIMUGyroNoise,
					rng.NormFloat64() * replayIMUGyroNoise,
					w + rng.NormFloat64()*replayIMUGyroNoise,
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
		gapStart := start.Add(cfg.Duration / 2)
		gapEnd := gapStart.Add(cfg.PoseGap)
		for t := start; t.Before(start.Add(cfg.Duration)); t = t.Add(dt) {
			if cfg.PoseGap > 0 && !t.Before(gapStart) && t.Before(gapEnd) {
				continue // 缺测窗口
			}
			if rng.Float64() < cfg.DropProb {
				continue
			}
			pose := poseAt(t)
			// 各路时钟漂移：按该路漂移曲线换算原始时间戳。
			stamp := rawFor(SensorAbsolutePose, t)
			o := Observation{
				Kind:  SensorAbsolutePose,
				Stamp: stamp,
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
