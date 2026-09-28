package fusion

import (
	"math"
	"testing"
	"time"
)

// alignReplayConfig 构造一路 60s、位姿/里程计带时变漂移的回放。
func alignReplayConfig(seed int64) ReplayConfig {
	return ReplayConfig{
		Seed: seed, Duration: 60 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.03, OutlierProb: 0.02, OutlierScale: 5,
		ReorderWindow: 16,
		PoseGap:       3 * time.Second,
		Clocks: map[SensorKind]ClockProfile{
			// 位姿路：起步即有 +120ms 固定偏移（读数快），35s 处短暂
			// 变快 5s 后回零，覆盖“偏移 + 运行中速率变化”。
			SensorAbsolutePose: {
				{At: 0, Offset: 120 * time.Millisecond, Skew: 0},
				{At: 35 * time.Second, Offset: 120 * time.Millisecond, Skew: 400e-6},
				{At: 40 * time.Second, Offset: 122 * time.Millisecond, Skew: 0},
			},
			// 里程计路：起步 −80ms，20s 后短暂变慢。
			SensorOdometry: {
				{At: 0, Offset: -80 * time.Millisecond, Skew: 0},
				{At: 20 * time.Second, Offset: -80 * time.Millisecond, Skew: -300e-6},
				{At: 26 * time.Second, Offset: -82 * time.Millisecond, Skew: 0},
			},
		},
	}
}

func runReplay(t *testing.T, obs []Observation, cfg Config) *Engine {
	t.Helper()
	eng := NewEngine(cfg)
	const batchSize = 64
	for i := 0; i < len(obs); i += batchSize {
		end := i + batchSize
		if end > len(obs) {
			end = len(obs)
		}
		eng.Ingest(obs[i:end])
	}
	var end time.Time
	for _, o := range obs {
		// 用 IMU 时间轴（统一时间基准）推进。
		if o.Kind == SensorIMU && o.Stamp.After(end) {
			end = o.Stamp
		}
	}
	eng.AdvanceTo(end)
	return eng
}

// TestAlignmentTimeVaryingDrift 时变、异向漂移下融合结果不得产生明显
// 时间错位或位姿偏差；且对齐模型最终必须处于可信 active 状态。
func TestAlignmentTimeVaryingDrift(t *testing.T) {
	cfg := alignReplayConfig(7)
	obs := GenerateReplay(cfg)
	t.Logf("回放规模: %d 条 / %v", len(obs), cfg.Duration)

	driftCfg := replayTestConfig()
	driftCfg.AlignMinSamples = 20
	driftCfg.AlignConfirmSamples = 6
	driftCfg.AlignActivateAt = 40 * time.Millisecond
	driftCfg.AlignMeasStd = 50 * time.Millisecond
	driftCfg.AlignForgetting = 0.995
	driftCfg.AlignRequireCalibration = true
	driftCfg.InitFallbackAfter = 3 * time.Second
	eng := runReplay(t, obs, driftCfg)
	st := eng.AlignStatuses()
	t.Logf("对齐状态: imu=%s odom=%s(off=%v,skew=%.0fppm) pose=%s(off=%v,skew=%.0fppm)",
		st[SensorIMU].Phase,
		st[SensorOdometry].Phase, st[SensorOdometry].Offset, st[SensorOdometry].Skew*1e6,
		st[SensorAbsolutePose].Phase, st[SensorAbsolutePose].Offset,
		st[SensorAbsolutePose].Skew*1e6)
	if st[SensorAbsolutePose].Phase != "active" {
		t.Fatalf("漂移位姿路应在线辨识为 active, got %s", st[SensorAbsolutePose].Phase)
	}
	if !st[SensorOdometry].Trusted {
		t.Fatal("里程计路换算必须可信（恒等先验或已激活）")
	}

	s, cov, ok := eng.Snapshot()
	if !ok {
		t.Fatal("引擎未初始化")
	}
	// 统一时间轴以 IMU（无漂移）为基准：真值按状态时刻直接评估。
	traj := DefaultTrajectory(testEpoch, cfg.Duration)
	truth := traj.TruePoseAt(s.Stamp)
	posErr := vecNorm(vecSub(s.Position, truth.Position))
	rotErr := vecNorm(QuatToRotVec(QuatMul(QuatConj(s.Rotation), truth.Rotation)))
	t.Logf("末端误差: pos=%.3fm rot=%.4frad (状态时刻=%v)", posErr, rotErr, s.Stamp)
	if posErr > 2.0 {
		t.Fatalf("时变漂移下位置误差超限: %.3fm", posErr)
	}
	if rotErr > 0.2 {
		t.Fatalf("时变漂移下姿态误差超限: %.4frad", rotErr)
	}
	if !cov.IsPositiveDefinite() {
		t.Fatal("协方差失去正定性")
	}

	// 对比关闭对齐的参照运行：开启对齐必须显著更优（证明确实在纠正时间错位）。
	offCfg := replayTestConfig()
	b := false
	offCfg.AlignEnabled = &b
	engOff := runReplay(t, obs, offCfg)
	sOff, _, _ := engOff.Snapshot()
	truthOff := traj.TruePoseAt(sOff.Stamp)
	posErrOff := vecNorm(vecSub(sOff.Position, truthOff.Position))
	t.Logf("对照(无对齐)位置误差=%.3fm, 对齐后=%.3fm", posErrOff, posErr)
	if posErr >= posErrOff {
		t.Fatalf("在线对齐未改善时间错位: aligned %.3f vs raw %.3f", posErr, posErrOff)
	}
}

// TestAlignmentDeterministic 相同时变漂移输入重复运行，姿态/协方差严格一致。
func TestAlignmentDeterministic(t *testing.T) {
	obs := GenerateReplay(alignReplayConfig(11))
	detCfg := replayTestConfig()
	detCfg.AlignMinSamples = 20
	detCfg.AlignConfirmSamples = 6
	detCfg.AlignActivateAt = 40 * time.Millisecond
	detCfg.AlignMeasStd = 50 * time.Millisecond
	detCfg.AlignForgetting = 0.995
	run := func() (State, Mat, []Decision) {
		eng := runReplay(t, obs, detCfg)
		s, c, _ := eng.Snapshot()
		return s, c, eng.Decisions()
	}
	s1, c1, d1 := run()
	s2, c2, d2 := run()
	// 逐字段严格比较（不经四元数→旋转向量的非线性舍入）。
	if s1.Position != s2.Position || s1.Velocity != s2.Velocity || s1.Rotation != s2.Rotation ||
		!s1.Stamp.Equal(s2.Stamp) {
		t.Fatalf("对齐后重复运行状态不一致:\n%+v\n%+v", s1, s2)
	}
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if c1.At(i, j) != c2.At(i, j) {
				t.Fatalf("协方差重复运行不一致 (%d,%d): %v vs %v", i, j, c1.At(i, j), c2.At(i, j))
			}
		}
	}
	if len(d1) != len(d2) {
		t.Fatalf("判定序列长度不一致: %d vs %d", len(d1), len(d2))
	}
	for i := range d1 {
		if d1[i].Reason != d2[i].Reason || d1[i].Accepted != d2[i].Accepted {
			t.Fatalf("判定序列第 %d 条不一致", i)
		}
	}
	t.Logf("重复运行严格一致: %d 条判定, 末态 pos=%v", len(d1), s1.Position)
}

// TestAlignmentZeroDriftBitIdentical 无漂移输入：开启对齐与关闭对齐
// 的融合结果必须逐位一致（恒等快路径，不靠放宽精度）。
func TestAlignmentZeroDriftBitIdentical(t *testing.T) {
	cfg := ReplayConfig{
		Seed: 3, Duration: 30 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.03, OutlierProb: 0.02, OutlierScale: 5,
		ReorderWindow: 16, PoseGap: 2 * time.Second,
	}
	obs := GenerateReplay(cfg)

	onEng := runReplay(t, obs, replayTestConfig())
	offCfg := replayTestConfig()
	b := false
	offCfg.AlignEnabled = &b
	offEng := runReplay(t, obs, offCfg)

	s1, c1, _ := onEng.Snapshot()
	s2, c2, _ := offEng.Snapshot()
	if s1.Position != s2.Position || s1.Velocity != s2.Velocity || s1.Rotation != s2.Rotation ||
		!s1.Stamp.Equal(s2.Stamp) {
		t.Fatalf("无漂移输入结果非逐位一致:\n on =%+v\n off=%+v", s1, s2)
	}
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if c1.At(i, j) != c2.At(i, j) {
				t.Fatalf("无漂移协方差非逐位一致 (%d,%d): %.18g vs %.18g",
					i, j, c1.At(i, j), c2.At(i, j))
			}
		}
	}
	for k, st := range onEng.AlignStatuses() {
		if st.Mapped {
			t.Fatalf("无漂移路 %s 不应启用估计映射", k)
		}
	}
	t.Logf("无漂移逐位一致: pos=%v stamp=%v", s1.Position, s1.Stamp)
}

// TestAlignmentMonotonicAfterModelChange 模型切换后状态时间不得回退，
// 已处理时间段不得被重新消费：构造早到观测在模型生效后落入过去。
func TestAlignmentMonotonicAfterModelChange(t *testing.T) {
	cfg := alignTestConfig()
	cfg.AlignMinSamples = 8
	cfg.AlignConfirmSamples = 4
	cfg.AlignActivateAt = 20 * time.Millisecond
	cfg.AlignMeasStd = 5 * time.Millisecond
	eng := NewEngine(cfg)
	eng.Ingest(makeInitObs(testEpoch))

	// 喂足以激活位姿路的时钟样本（通过真实观测的创新学习）较难直接构造，
	// 这里改为验证语义不变量：模型激活后，换算到过去的观测被 regress/stale 拒绝。
	a := eng.align
	m := a.ensure(SensorAbsolutePose, testEpoch, false)
	rl := a.rls[SensorAbsolutePose]
	rl.theta = [2]float64{-0.1, 1} // 位姿读数快 100ms
	m.phase = phaseActive
	m.n, m.confirm = 20, 20

	// 当前状态时间约 testEpoch+20ms。来一条读数 t+100ms 的位姿观测，
	// 换算后 ≈ t（当前时刻附近），先正常推进。
	obs1 := Observation{Kind: SensorAbsolutePose, Stamp: testEpoch.Add(120 * time.Millisecond), Seq: 900,
		Pose:    Pose{Position: Vec3{}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	ds := eng.Ingest([]Observation{obs1})
	t.Logf("模型生效后观测1: accepted=%v reason=%v detail=%s",
		ds[0].Accepted, ds[0].Reason, ds[0].Detail)

	before, _, _ := eng.Snapshot()
	// 再来读数更早的观测：换算后落在状态时间之前 → 必须拒绝且状态不变。
	obs2 := Observation{Kind: SensorAbsolutePose, Stamp: testEpoch.Add(60 * time.Millisecond), Seq: 901,
		Pose:    Pose{Position: Vec3{X: 77}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	ds = eng.Ingest([]Observation{obs2})
	t.Logf("落到过去的观测: accepted=%v reason=%v detail=%s",
		ds[0].Accepted, ds[0].Reason, ds[0].Detail)
	if ds[0].Accepted {
		t.Fatal("模型更新后落到过去的观测不得被采用")
	}
	if ds[0].Reason != ReasonTimestampRegress && ds[0].Reason != ReasonStale {
		t.Fatalf("应为 regress/stale, got %v", ds[0].Reason)
	}
	after, _, _ := eng.Snapshot()
	if !statesEqual(before, after, 0) {
		t.Fatal("拒绝观测后状态发生了变化")
	}
	if after.Stamp.Before(before.Stamp) {
		t.Fatal("状态时间发生回退")
	}
}

// TestAlignmentCalibratingRejectedDistinct 初始化后才出现的新路，
// 样本不足期间必须以 align_calibrating 暂不采用，且不污染状态。
func TestAlignmentCalibratingRejectedDistinct(t *testing.T) {
	eng := NewEngine(testConfig())
	// 仅用 IMU+里程计初始化（无位姿），绝对位姿路成为“后出现”的路。
	in := []Observation{
		{Kind: SensorIMU, Stamp: testEpoch, Seq: 0, Accel: Vec3{Z: 9.81}},
		{Kind: SensorOdometry, Stamp: testEpoch.Add(10 * time.Millisecond), Seq: 1,
			LinVel: Vec3{X: 1}, CovDiag: [9]float64{0.01, 0.01, 0.01}},
	}
	for i := 2; i < 9; i++ {
		in = append(in, Observation{Kind: SensorIMU,
			Stamp: testEpoch.Add(time.Duration(i*10) * time.Millisecond), Seq: uint64(i),
			Accel: Vec3{Z: 9.81}})
	}
	eng.Ingest(in)
	sBefore, _, ok := eng.Snapshot()
	if !ok {
		t.Fatal("应已兜底初始化")
	}
	pose := Observation{Kind: SensorAbsolutePose, Stamp: testEpoch.Add(200 * time.Millisecond), Seq: 50,
		Pose:    Pose{Position: Vec3{X: 42}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	ds := eng.Ingest([]Observation{pose})
	t.Logf("冷启动位姿: accepted=%v reason=%v detail=%s",
		ds[0].Accepted, ds[0].Reason, ds[0].Detail)
	if ds[0].Accepted || ds[0].Reason != ReasonAlignCalibrating {
		t.Fatalf("对齐样本不足必须 align_calibrating, got accepted=%v reason=%v",
			ds[0].Accepted, ds[0].Reason)
	}
	sAfter, _, _ := eng.Snapshot()
	if !statesEqual(sBefore, sAfter, 0) {
		t.Fatal("校准期观测不得污染状态")
	}
}

// 确保 math 包被使用（误差阈值常量集中处）。
var _ = math.Abs
