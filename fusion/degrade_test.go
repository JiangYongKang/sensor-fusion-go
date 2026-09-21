package fusion

import (
	"sync"
	"testing"
	"time"
)

// TestSensorDegradation 某路观测长时间缺失时进入降级；
// 降级期间到达的观测被噪声膨胀处理且判定原因可区分。
func TestSensorDegradation(t *testing.T) {
	t0 := testEpoch
	cfg := testConfig()
	cfg.DegradeAfter = 200 * time.Millisecond
	eng := NewEngine(cfg)
	eng.Ingest(makeInitObs(t0))

	// 推进到 1s 后（期间无任何新观测），绝对位姿路应进入降级。
	eng.AdvanceTo(t0.Add(1 * time.Second))
	deg := eng.Degraded()
	t.Logf("1s 无观测后降级状态: %v", deg)
	if !deg[SensorAbsolutePose] || !deg[SensorOdometry] {
		t.Fatalf("缺测路应进入降级: %v", deg)
	}

	// 降级后到达的绝对位姿观测：被接受但标记为降级恢复（噪声膨胀）。
	late := Observation{Kind: SensorAbsolutePose, Stamp: t0.Add(1100 * time.Millisecond), Seq: 300,
		Pose:    Pose{Position: Vec3{X: 0.5}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	eng.Ingest([]Observation{late})
	ds := eng.Decisions()
	last := ds[len(ds)-1]
	t.Logf("降级恢复观测: accepted=%v reason=%v detail=%s", last.Accepted, last.Reason, last.Detail)
	if !last.Accepted || last.Reason != ReasonSensorDegraded {
		t.Fatalf("降级恢复观测应被膨胀接受且原因可区分, got accepted=%v reason=%v",
			last.Accepted, last.Reason)
	}
}

// TestStaleNotUsedAsFresh 陈旧观测不得当作最新数据参与更新。
func TestStaleNotUsedAsFresh(t *testing.T) {
	t0 := testEpoch
	eng := NewEngine(testConfig())
	eng.Ingest(makeInitObs(t0))
	eng.AdvanceTo(t0.Add(2 * time.Second))
	sBefore, _, _ := eng.Snapshot()

	// 远落后于当前状态时间的观测（超出 StaleAfter）。
	stale := Observation{Kind: SensorAbsolutePose, Stamp: t0.Add(1 * time.Second), Seq: 400,
		Pose:    Pose{Position: Vec3{X: 99}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	ds := eng.Ingest([]Observation{stale})
	last := ds[len(ds)-1]
	t.Logf("陈旧观测: accepted=%v reason=%v detail=%s", last.Accepted, last.Reason, last.Detail)
	if last.Accepted || last.Reason != ReasonStale {
		t.Fatalf("陈旧观测应被拒绝, got accepted=%v reason=%v", last.Accepted, last.Reason)
	}
	sAfter, _, _ := eng.Snapshot()
	if !statesEqual(sBefore, sAfter, 1e-12) {
		t.Fatal("陈旧观测参与了状态更新")
	}
}

// TestConcurrentConsistency 预测、更新与快照读取并发交错时
// 不得出现半更新状态或数据竞争（配合 -race 运行）。
func TestConcurrentConsistency(t *testing.T) {
	t0 := testEpoch
	eng := NewEngine(testConfig())
	eng.Ingest(makeInitObs(t0))

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// 写入方：持续摄入 IMU/里程计/绝对位姿。
	for w := 0; w < 3; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			base := int64(id) * 100000
			for i := 0; i < 200; i++ {
				ts := t0.Add(time.Duration(base+int64(i)*10) * time.Millisecond)
				obs := []Observation{
					{Kind: SensorIMU, Stamp: ts, Seq: uint64(base + int64(i)*3),
						Accel: Vec3{Z: 9.81}},
					{Kind: SensorOdometry, Stamp: ts.Add(time.Millisecond), Seq: uint64(base + int64(i)*3 + 1),
						LinVel:  Vec3{X: 1},
						CovDiag: [9]float64{0.01, 0.01, 0.01}},
					{Kind: SensorAbsolutePose, Stamp: ts.Add(2 * time.Millisecond), Seq: uint64(base + int64(i)*3 + 2),
						Pose:    Pose{Position: Vec3{X: float64(i) * 0.01}, Rotation: QuatIdentity()},
						CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}},
				}
				eng.Ingest(obs)
			}
		}(w)
	}

	// 读取方：持续快照与推进。
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			default:
				s, cov, ok := eng.Snapshot()
				if ok {
					// 快照必须自洽：四元数保持单位范数，协方差保持正定。
					n := s.Rotation.W*s.Rotation.W + s.Rotation.X*s.Rotation.X +
						s.Rotation.Y*s.Rotation.Y + s.Rotation.Z*s.Rotation.Z
					if n < 0.99 || n > 1.01 {
						t.Errorf("快照中四元数退化: |q|^2=%v", n)
						return
					}
					if !cov.IsPositiveDefinite() {
						t.Error("快照中协方差失去正定性")
						return
					}
				}
			}
		}
	}()

	wg.Wait()
	close(stop)
	<-done
	t.Logf("并发测试完成, 判定日志条数=%d", len(eng.Decisions()))
}

// TestRepeatRunDeterministic 相同输入序列重复运行必须得到一致结果。
func TestRepeatRunDeterministic(t *testing.T) {
	t0 := testEpoch
	run := func() State {
		eng := NewEngine(testConfig())
		eng.Ingest(makeInitObs(t0))
		for i := 0; i < 50; i++ {
			ts := t0.Add(time.Duration(30+i*20) * time.Millisecond)
			eng.Ingest([]Observation{
				{Kind: SensorIMU, Stamp: ts, Seq: uint64(i), Accel: Vec3{Z: 9.81}},
				{Kind: SensorOdometry, Stamp: ts.Add(5 * time.Millisecond), Seq: uint64(1000 + i),
					LinVel:  Vec3{X: 1},
					CovDiag: [9]float64{0.01, 0.01, 0.01}},
			})
		}
		s, _, _ := eng.Snapshot()
		return s
	}
	a := run()
	b := run()
	t.Logf("运行A末态 pos=%+v vel=%+v", a.Position, a.Velocity)
	if !statesEqual(a, b, 0) {
		t.Fatalf("重复运行结果不一致: A=%+v B=%+v", a.Position, b.Position)
	}
	if a.Rotation != b.Rotation {
		t.Fatalf("重复运行姿态不一致: A=%+v B=%+v", a.Rotation, b.Rotation)
	}
}
