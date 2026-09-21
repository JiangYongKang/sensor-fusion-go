package fusion

import (
	"math"
	"testing"
	"time"
)

// replayTestConfig 回放测试配置：与回放噪声特性匹配。
func replayTestConfig() Config {
	cfg := testConfig()
	cfg.ProcessNoise = [9]float64{
		0.001, 0.001, 0.001, // δp
		0.05, 0.05, 0.05, // δv
		0.001, 0.001, 0.001, // δθ
	}
	cfg.DegradeAfter = 3 * time.Second
	cfg.MaxClockDrift = 10 * time.Second
	return cfg
}

// TestReplayDeterministic 回放生成器可复现：相同配置生成相同序列。
func TestReplayDeterministic(t *testing.T) {
	cfg := ReplayConfig{
		Seed: 42, Duration: 10 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.05, OutlierProb: 0.02, OutlierScale: 5,
		ReorderWindow: 8, ClockDrift: 100 * time.Millisecond,
		PoseGap: 2 * time.Second,
	}
	a := GenerateReplay(cfg)
	b := GenerateReplay(cfg)
	if len(a) != len(b) || len(a) == 0 {
		t.Fatalf("回放长度不一致或为空: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("回放第 %d 条不一致", i)
		}
	}
	t.Logf("回放生成: %d 条观测 (imu/odom/pose 混合, 乱序窗口=%d)",
		len(a), cfg.ReorderWindow)
}

// TestLongRunFusion 长时间运行：乱序+丢帧+异常+漂移+缺测全场景下，
// 估计结果保持有界误差、姿态不退化、协方差正定，且重复运行一致。
func TestLongRunFusion(t *testing.T) {
	cfg := ReplayConfig{
		Seed: 7, Duration: 60 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.05, OutlierProb: 0.02, OutlierScale: 5,
		ReorderWindow: 16, ClockDrift: 50 * time.Millisecond,
		PoseGap: 5 * time.Second,
	}
	obs := GenerateReplay(cfg)
	t.Logf("回放规模: %d 条观测 / %v", len(obs), cfg.Duration)

	run := func() (State, Mat, map[RejectReason]int) {
		eng := NewEngine(replayTestConfig())
		// 分批摄入（每批 64 条），批内乱序由重排缓冲排序。
		const batchSize = 64
		for i := 0; i < len(obs); i += batchSize {
			end := i + batchSize
			if end > len(obs) {
				end = len(obs)
			}
			eng.Ingest(obs[i:end])
		}
		// 推进到回放最晚观测时刻，消费缓冲中剩余观测。
		end := obs[0].Stamp
		for _, o := range obs {
			if o.Stamp.After(end) {
				end = o.Stamp
			}
		}
		eng.AdvanceTo(end)
		s, cov, ok := eng.Snapshot()
		if !ok {
			t.Fatal("引擎未初始化")
		}
		// 统计判定原因分布。
		stat := make(map[RejectReason]int)
		for _, d := range eng.Decisions() {
			stat[d.Reason]++
		}
		return s, cov, stat
	}

	s1, cov1, stat1 := run()
	s2, _, stat2 := run()

	// 可复现性：两次运行结果一致（四元数经归一化，允许 1e-12 量级残差）。
	if !statesEqual(s1, s2, 1e-12) {
		t.Fatalf("重复运行不一致: %+v vs %+v", s1.Position, s2.Position)
	}
	if stat1[ReasonOutlierGate] != stat2[ReasonOutlierGate] {
		t.Fatal("异常门限判定次数在重复运行间不一致")
	}
	t.Logf("判定分布: %v", stat1)

	// 姿态数值稳定：四元数单位范数。
	q := s1.Rotation
	n := q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z
	if math.Abs(n-1) > 1e-9 {
		t.Fatalf("长时间运行后四元数退化: |q|^2=%v", n)
	}
	// 协方差正定性。
	if !cov1.IsPositiveDefinite() {
		t.Fatal("长时间运行后协方差失去正定性")
	}

	// 估计误差有界：与真值轨迹末端比较。
	// 注意回放末端存在时钟漂移，真值按观测时间戳评估。
	traj := DefaultTrajectory(testEpoch, cfg.Duration)
	endState := s1
	truth := traj.TruePoseAt(endState.Stamp)
	posErr := vecNorm(vecSub(endState.Position, truth.Position))
	rotErr := vecNorm(QuatToRotVec(QuatMul(QuatConj(endState.Rotation), truth.Rotation)))
	t.Logf("末端误差: pos=%.3fm rot=%.4frad (状态时刻=%v)", posErr, rotErr, endState.Stamp)
	if posErr > 2.0 {
		t.Fatalf("位置误差超限: %.3fm", posErr)
	}
	if rotErr > 0.2 {
		t.Fatalf("姿态误差超限: %.4frad", rotErr)
	}
	// 异常值必须被门限捕获（回放注入了 2% 异常）。
	if stat1[ReasonOutlierGate] == 0 {
		t.Fatal("异常值未被门限捕获")
	}
}

// TestClockDriftHandled 时钟漂移超界的观测被拒绝且原因可区分。
func TestClockDriftHandled(t *testing.T) {
	t0 := testEpoch
	cfg := testConfig()
	cfg.MaxClockDrift = 100 * time.Millisecond
	eng := NewEngine(cfg)
	eng.Ingest(makeInitObs(t0))

	drifted := Observation{Kind: SensorAbsolutePose, Stamp: t0.Add(10 * time.Second), Seq: 500,
		Pose:    Pose{Position: Vec3{X: 1}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	ds := eng.Ingest([]Observation{drifted})
	last := ds[len(ds)-1]
	t.Logf("漂移观测: accepted=%v reason=%v detail=%s", last.Accepted, last.Reason, last.Detail)
	if last.Accepted || last.Reason != ReasonTimeUnavailable {
		t.Fatalf("超界漂移观测应被拒绝, got accepted=%v reason=%v", last.Accepted, last.Reason)
	}
}

// TestReorderDelayWindow ReorderDelay>0 时窗口内乱序被正确重排。
func TestReorderDelayWindow(t *testing.T) {
	t0 := testEpoch
	cfg := testConfig()
	cfg.ReorderDelay = 100 * time.Millisecond
	eng := NewEngine(cfg)
	eng.Ingest(makeInitObs(t0))

	// 逐条乱序摄入（跨调用乱序，由 ReorderDelay 窗口保护）。
	seq := []Observation{
		{Kind: SensorOdometry, Stamp: t0.Add(50 * time.Millisecond), Seq: 10,
			LinVel: Vec3{X: 1}, CovDiag: [9]float64{0.01, 0.01, 0.01}},
		{Kind: SensorOdometry, Stamp: t0.Add(30 * time.Millisecond), Seq: 11, // 乱序：更早的后到
			LinVel: Vec3{X: 1}, CovDiag: [9]float64{0.01, 0.01, 0.01}},
		{Kind: SensorOdometry, Stamp: t0.Add(40 * time.Millisecond), Seq: 12,
			LinVel: Vec3{X: 1}, CovDiag: [9]float64{0.01, 0.01, 0.01}},
	}
	for _, o := range seq {
		eng.Ingest([]Observation{o})
	}
	eng.AdvanceTo(t0.Add(200 * time.Millisecond)) // 触发窗口内全部消费

	// 与顺序摄入的参照运行比较。
	ref := NewEngine(cfg)
	ref.Ingest(makeInitObs(t0))
	ref.Ingest([]Observation{seq[1], seq[2], seq[0]}) // 时间顺序
	ref.AdvanceTo(t0.Add(200 * time.Millisecond))

	sA, _, _ := eng.Snapshot()
	sB, _, _ := ref.Snapshot()
	t.Logf("乱序末态 pos=%v 顺序末态 pos=%v", sA.Position, sB.Position)
	if !statesEqual(sA, sB, 1e-12) {
		t.Fatalf("ReorderDelay 窗口内乱序未按时间顺序参与状态推进: %+v vs %+v",
			sA.Position, sB.Position)
	}
}
