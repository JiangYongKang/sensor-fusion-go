package fusion

import (
	"sync"
	"testing"
	"time"
)

// TestBudgetBoundaryThreeReasons 三类输入边界必须返回三个可区分原因，
// 且在产生任何状态变更前整批拒绝（拒绝后引擎可继续处理后续批次）。
func TestBudgetBoundaryThreeReasons(t *testing.T) {
	// 1) 单批规模超限：limit_exceeded。
	cfg1 := testConfig()
	cfg1.Limits.MaxBatchSize = 4
	eng := NewEngine(cfg1)
	eng.Ingest(makeInitObs(testEpoch))
	big := make([]Observation, 5)
	for i := range big {
		big[i] = Observation{Kind: SensorIMU,
			Stamp: testEpoch.Add(time.Duration(300+i*10) * time.Millisecond),
			Seq:   uint64(200 + i), Accel: Vec3{Z: 9.81}}
	}
	before, _, _ := eng.Snapshot()
	ds := eng.Ingest(big)
	if ds[0].Reason != ReasonBatchTooLarge {
		t.Fatalf("单批超限应 limit_exceeded, got %v", ds[0].Reason)
	}
	after, _, _ := eng.Snapshot()
	if !statesEqual(before, after, 1e-12) {
		t.Fatal("整批规模超限不得产生部分状态更新")
	}

	// 2) 预算不足（确定性成本模型）：budget_exceeded，同样无部分更新。
	budgetCfg := testConfig()
	budgetCfg.Limits.MaxProcessTime = 1 // 微秒：64 条观测成本远超
	eng2 := NewEngine(budgetCfg)
	eng2.Ingest(makeInitObs(testEpoch))
	batch := make([]Observation, 64)
	for i := range batch {
		batch[i] = Observation{Kind: SensorIMU,
			Stamp: testEpoch.Add(time.Duration(400+i*10) * time.Millisecond),
			Seq:   uint64(300 + i), Accel: Vec3{Z: 9.81}}
	}
	before2, _, _ := eng2.Snapshot()
	ds2 := eng2.Ingest(batch)
	if ds2[0].Reason != ReasonBudgetExceeded {
		t.Fatalf("预算不足应 budget_exceeded, got %v", ds2[0].Reason)
	}
	after2, _, _ := eng2.Snapshot()
	if !statesEqual(before2, after2, 1e-12) {
		t.Fatal("预算拒绝不得产生部分更新")
	}

	// 3) 积压过多：backlog_full，无部分更新；拒绝后仍能继续处理小批。
	backCfg := testConfig()
	backCfg.Limits.MaxPendingObs = 6
	eng3 := NewEngine(backCfg)
	eng3.Ingest(makeInitObs(testEpoch))
	// 用 ReorderDelay 让观测滞留缓冲（窗口内需保留）。
	eng3.cfg.ReorderDelay = 2 * time.Second
	hold := make([]Observation, 8)
	for i := range hold {
		hold[i] = Observation{Kind: SensorIMU,
			Stamp: testEpoch.Add(time.Duration(500+i*10) * time.Millisecond),
			Seq:   uint64(400 + i), Accel: Vec3{Z: 9.81}}
	}
	before3, _, _ := eng3.Snapshot()
	sawBacklog := false
	for _, d := range eng3.Ingest(hold) {
		if d.Reason == ReasonBacklogFull {
			sawBacklog = true
		}
	}
	if !sawBacklog {
		t.Fatal("缓冲打满应产生 backlog_full")
	}
	after3, _, _ := eng3.Snapshot()
	if !statesEqual(before3, after3, 1e-12) {
		t.Fatal("积压拒绝不得产生部分更新")
	}
}

// TestBudgetRejectedBatchResumeEngine 被边界拒绝的批次不留残更新，
// 引擎随后能继续正常摄入并推进。
func TestBudgetRejectedBatchResumeEngine(t *testing.T) {
	c := testConfig()
	c.Limits.MaxBatchSize = 4
	eng := NewEngine(c)
	eng.Ingest(makeInitObs(testEpoch))
	bad := make([]Observation, 5)
	for i := range bad {
		bad[i] = Observation{Kind: SensorIMU,
			Stamp: testEpoch.Add(time.Duration(300+i*10) * time.Millisecond),
			Seq:   uint64(500 + i), Accel: Vec3{Z: 9.81}}
	}
	eng.Ingest(bad) // 被规模拒绝
	good := Observation{Kind: SensorIMU,
		Stamp: testEpoch.Add(350 * time.Millisecond), Seq: 600, Accel: Vec3{Z: 9.81}}
	ds := eng.Ingest([]Observation{good})
	if len(ds) != 1 || !ds[0].Accepted {
		t.Fatalf("拒绝批次后引擎应继续可用, got %+v", ds)
	}
}

// TestOutOfRangeDistinctReasons 漂移超界与样本不足必须是两个不同原因。
func TestOutOfRangeDistinctReasons(t *testing.T) {
	// 样本不足：初始化后才出现的新路。
	eng := NewEngine(testConfig())
	in := []Observation{{Kind: SensorIMU, Stamp: testEpoch, Seq: 0, Accel: Vec3{Z: 9.81}}}
	for i := 1; i < 9; i++ {
		in = append(in, Observation{Kind: SensorIMU,
			Stamp: testEpoch.Add(time.Duration(i*10) * time.Millisecond),
			Seq:   uint64(i), Accel: Vec3{Z: 9.81}})
	}
	eng.Ingest(in)
	late := Observation{Kind: SensorAbsolutePose, Stamp: testEpoch.Add(200 * time.Millisecond), Seq: 700,
		Pose:    Pose{Position: Vec3{}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	if d := eng.Ingest([]Observation{late})[0]; d.Reason != ReasonAlignCalibrating {
		t.Fatalf("样本不足应 align_calibrating, got %v", d.Reason)
	}

	// 漂移超界：观测时间偏离统一时间超过 MaxClockDrift。
	eng2 := NewEngine(testConfig())
	eng2.Ingest(makeInitObs(testEpoch))
	o := Observation{Kind: SensorAbsolutePose,
		Stamp: testEpoch.Add(10 * time.Second), Seq: 701,
		Pose:    Pose{Position: Vec3{X: 1}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	d2 := eng2.Ingest([]Observation{o})[0]
	if d2.Reason != ReasonTimeUnavailable {
		t.Fatalf("漂移超界应 time_unavailable, got %v", d2.Reason)
	}
	if ReasonAlignCalibrating == ReasonAlignOutOfRange ||
		ReasonAlignCalibrating == ReasonTimeUnavailable {
		t.Fatal("样本不足、漂移超界原因必须彼此可区分")
	}
}

// TestConcurrentPredictUpdateSnapshot 预测/更新/快照交错并发下不出现
// 数据竞争、状态不自洽（协方差始终正定、四元数单位范数）。
func TestConcurrentPredictUpdateSnapshot(t *testing.T) {
	eng := NewEngine(testConfig())
	eng.Ingest(makeInitObs(testEpoch))
	var wg sync.WaitGroup
	stop := make(chan struct{})
	// 写入方：持续摄入 IMU 与里程计。
	wg.Add(1)
	go func() {
		defer wg.Done()
		i := uint64(800)
		for {
			select {
			case <-stop:
				return
			default:
			}
			ts := testEpoch.Add(time.Duration(i) * 10 * time.Millisecond)
			eng.Ingest([]Observation{{Kind: SensorIMU, Stamp: ts, Seq: i,
				Accel: Vec3{Z: 9.81}, Gyro: Vec3{Z: 0.01}}})
			if i%5 == 0 {
				eng.Ingest([]Observation{{Kind: SensorOdometry, Stamp: ts, Seq: i + 1,
					LinVel: Vec3{X: 1, Y: 0.01}, CovDiag: [9]float64{0.01, 0.01, 0.01}}})
			}
			i += 2
		}
	}()
	// 推进方：偶发 AdvanceTo。
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 40; j++ {
			select {
			case <-stop:
				return
			default:
			}
			s, _, ok := eng.Snapshot()
			if ok {
				eng.AdvanceTo(s.Stamp.Add(20 * time.Millisecond))
			}
		}
	}()
	// 快照方：高并发读取，检查自洽性。
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				s, cov, ok := eng.Snapshot()
				if !ok {
					continue
				}
				q := s.Rotation
				n := q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z
				if n < 0.9 || n > 1.1 {
					t.Errorf("并发快照四元数退化: %v", n)
					return
				}
				if !cov.IsPositiveDefinite() {
					t.Errorf("并发快照协方差非正定")
					return
				}
			}
		}()
	}
	// 状态/对齐诊断读取方。
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 200; j++ {
			select {
			case <-stop:
				return
			default:
			}
			_ = eng.AlignStatuses()
			_ = eng.Stats()
			_ = eng.Degraded()
		}
	}()
	// 让写入方跑一会儿后停止。
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 100; j++ {
			time.Sleep(time.Millisecond)
		}
		close(stop)
	}()
	wg.Wait()
}
