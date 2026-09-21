package fusion

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// testConfig 测试用引擎配置。
func testConfig() Config {
	return Config{
		Limits: Limits{
			MaxPendingObs:  512,
			MaxBatchSize:   128,
			MaxProcessTime: time.Second,
		},
		ProcessNoise:  [9]float64{0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01},
		GateChi2:      16.0,
		StaleAfter:    500 * time.Millisecond,
		DegradeAfter:  2 * time.Second,
		MaxClockDrift: 5 * time.Second,
		InitWindow:    3,
	}
}

var testEpoch = time.Unix(1_700_000_000, 0)

// makeInitObs 构造初始化观测（绝对位姿 + 里程计 + IMU）。
func makeInitObs(t0 time.Time) []Observation {
	return []Observation{
		{Kind: SensorAbsolutePose, Stamp: t0, Seq: 0,
			Pose:    Pose{Position: Vec3{}, Rotation: QuatIdentity()},
			CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}},
		{Kind: SensorOdometry, Stamp: t0.Add(10 * time.Millisecond), Seq: 1,
			LinVel:  Vec3{X: 1},
			CovDiag: [9]float64{0.01, 0.01, 0.01}},
		{Kind: SensorIMU, Stamp: t0.Add(20 * time.Millisecond), Seq: 2,
			Accel: Vec3{Z: 9.81}},
	}
}

func statesEqual(a, b State, tol float64) bool {
	if vecNorm(vecSub(a.Position, b.Position)) > tol {
		return false
	}
	if vecNorm(vecSub(a.Velocity, b.Velocity)) > tol {
		return false
	}
	dq := QuatMul(QuatConj(a.Rotation), b.Rotation)
	return vecNorm(QuatToRotVec(dq)) <= tol
}

// TestOutOfOrderDeterministic 乱序与顺序摄入同一批观测，最终状态必须一致。
func TestOutOfOrderDeterministic(t *testing.T) {
	t0 := testEpoch
	init := makeInitObs(t0)

	// 构造 20 条后续观测（IMU + 里程计混合）。
	var batch []Observation
	for i := 0; i < 20; i++ {
		ts := t0.Add(time.Duration(30+i*10) * time.Millisecond)
		if i%2 == 0 {
			batch = append(batch, Observation{Kind: SensorIMU, Stamp: ts, Seq: uint64(10 + i),
				Accel: Vec3{Z: 9.81}})
		} else {
			batch = append(batch, Observation{Kind: SensorOdometry, Stamp: ts, Seq: uint64(10 + i),
				LinVel:  Vec3{X: 1},
				CovDiag: [9]float64{0.01, 0.01, 0.01}})
		}
	}

	// 运行 A：顺序摄入。
	engA := NewEngine(testConfig())
	engA.Ingest(init)
	engA.Ingest(batch)
	sA, cA, okA := engA.Snapshot()

	// 运行 B：确定性打乱后摄入（相同集合，不同到达顺序）。
	shuffled := make([]Observation, len(batch))
	copy(shuffled, batch)
	rand.New(rand.NewSource(7)).Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	t.Logf("乱序到达顺序(seq): %v", seqsOf(shuffled))
	engB := NewEngine(testConfig())
	engB.Ingest(init)
	engB.Ingest(shuffled)
	sB, cB, okB := engB.Snapshot()

	if !okA || !okB {
		t.Fatal("engine not initialized")
	}
	t.Logf("顺序末态 pos=%v 乱序末态 pos=%v", sA.Position, sB.Position)
	if !statesEqual(sA, sB, 1e-9) {
		t.Fatalf("状态随到达顺序变化: A=%+v B=%+v", sA, sB)
	}
	for i := 0; i < 9; i++ {
		if math.Abs(cA.At(i, i)-cB.At(i, i)) > 1e-12 {
			t.Fatalf("协方差随到达顺序变化 at diag %d: %v vs %v", i, cA.At(i, i), cB.At(i, i))
		}
	}
}

func seqsOf(obs []Observation) []uint64 {
	out := make([]uint64, len(obs))
	for i, o := range obs {
		out[i] = o.Seq
	}
	return out
}

// TestDuplicateAndRegress 重复与倒退时间戳必须被确定且可区分地拒绝。
func TestDuplicateAndRegress(t *testing.T) {
	t0 := testEpoch
	eng := NewEngine(testConfig())
	eng.Ingest(makeInitObs(t0))

	pose := Observation{Kind: SensorAbsolutePose, Stamp: t0.Add(100 * time.Millisecond), Seq: 100,
		Pose:    Pose{Position: Vec3{X: 0.1}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	ds := eng.Ingest([]Observation{pose})
	last := ds[len(ds)-1]
	t.Logf("首次摄入: accepted=%v reason=%v", last.Accepted, last.Reason)
	if !last.Accepted {
		t.Fatalf("首次摄入应被接受, got %v", last.Reason)
	}

	// 完全相同的观测再次到达 → 重复。
	ds = eng.Ingest([]Observation{pose})
	last = ds[len(ds)-1]
	t.Logf("重复摄入: accepted=%v reason=%v detail=%s", last.Accepted, last.Reason, last.Detail)
	if last.Accepted || last.Reason != ReasonDuplicate {
		t.Fatalf("重复观测应判定为 duplicate, got accepted=%v reason=%v", last.Accepted, last.Reason)
	}

	// 时间戳倒退的观测 → 可区分原因（stale 或 regress）。
	old := Observation{Kind: SensorAbsolutePose, Stamp: t0.Add(50 * time.Millisecond), Seq: 101,
		Pose:    Pose{Position: Vec3{X: 0.05}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}
	ds = eng.Ingest([]Observation{old})
	last = ds[len(ds)-1]
	t.Logf("倒退摄入: accepted=%v reason=%v detail=%s", last.Accepted, last.Reason, last.Detail)
	if last.Accepted {
		t.Fatal("倒退观测不得被接受")
	}
	if last.Reason != ReasonTimestampRegress && last.Reason != ReasonStale {
		t.Fatalf("倒退观测原因应为 regress/stale, got %v", last.Reason)
	}
}

// TestOutlierRejected 异常观测不得污染估计状态：
// 含异常值的运行与不含异常值的参照运行必须得到一致状态。
func TestOutlierRejected(t *testing.T) {
	t0 := testEpoch
	outlier := Observation{Kind: SensorAbsolutePose, Stamp: t0.Add(100 * time.Millisecond), Seq: 50,
		Pose:    Pose{Position: Vec3{X: 1000, Y: 1000}, Rotation: QuatIdentity()},
		CovDiag: [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}}

	// 运行 A：摄入异常观测。
	engA := NewEngine(testConfig())
	engA.Ingest(makeInitObs(t0))
	engA.Ingest([]Observation{outlier})
	// 门限判定发生在消费阶段，从判定日志取最后一条。
	dsA := engA.Decisions()
	last := dsA[len(dsA)-1]
	t.Logf("异常观测: accepted=%v reason=%v maha^2=%.2f gate=%.1f",
		last.Accepted, last.Reason, last.Maha, testConfig().GateChi2)
	if last.Accepted || last.Reason != ReasonOutlierGate {
		t.Fatalf("异常值应被门限拒绝, got accepted=%v reason=%v", last.Accepted, last.Reason)
	}

	// 运行 B：参照，不含异常观测，推进到相同时刻。
	engB := NewEngine(testConfig())
	engB.Ingest(makeInitObs(t0))
	engB.AdvanceTo(t0.Add(100 * time.Millisecond))

	sA, _, _ := engA.Snapshot()
	sB, _, _ := engB.Snapshot()
	t.Logf("含异常运行末态 pos=%v 参照末态 pos=%v", sA.Position, sB.Position)
	if !statesEqual(sA, sB, 1e-12) {
		t.Fatalf("异常观测污染了状态: A=%+v B=%+v", sA.Position, sB.Position)
	}
}

// TestBatchLimitRejected 超限批次整批拒绝，不残留部分更新。
func TestBatchLimitRejected(t *testing.T) {
	t0 := testEpoch
	cfg := testConfig()
	cfg.Limits.MaxBatchSize = 4
	eng := NewEngine(cfg)
	eng.Ingest(makeInitObs(t0))
	sBefore, _, _ := eng.Snapshot()

	big := make([]Observation, 5) // 超过 MaxBatchSize=4
	for i := range big {
		big[i] = Observation{Kind: SensorIMU, Stamp: t0.Add(time.Duration(200+i*10) * time.Millisecond),
			Seq: uint64(200 + i), Accel: Vec3{Z: 9.81}}
	}
	ds := eng.Ingest(big)
	for i, d := range ds {
		t.Logf("批内观测 %d: accepted=%v reason=%v", i, d.Accepted, d.Reason)
		if d.Accepted || d.Reason != ReasonLimitExceeded {
			t.Fatalf("超限批次应整批拒绝, obs %d got accepted=%v reason=%v", i, d.Accepted, d.Reason)
		}
	}
	sAfter, _, _ := eng.Snapshot()
	if !statesEqual(sBefore, sAfter, 1e-12) {
		t.Fatal("超限批次造成了部分更新")
	}
}
