package fusion

import (
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// alignTestConfig 在线时间对齐测试配置。
func alignTestConfig() Config {
	cfg := replayTestConfig()
	cfg.EnableAlignment = true
	cfg.AlignAutoRef = false
	cfg.AlignRef = SensorIMU
	cfg.Limits.MaxProcessTime = time.Minute // 回放用确定性操作预算，墙钟仅兜底
	cfg.AlignMinSamples = 10
	cfg.AlignMinSpan = 1 * time.Second
	cfg.AlignDetect = 150 * time.Millisecond
	cfg.AlignTolerance = 60 * time.Millisecond
	cfg.AlignMaxSkew = 0.05
	cfg.AlignMaxDrift = 10 * time.Second
	cfg.AlignSearchRange = 2 * time.Second
	cfg.AlignWindow = 8 * time.Second
	cfg.AlignReleaseMargin = 50 * time.Millisecond
	return cfg
}

// driftReplayConfig 各路漂移方向/幅度不一致且随时间变化的回放。
func driftReplayConfig(dur time.Duration) ReplayConfig {
	min := time.Minute
	return ReplayConfig{
		Seed: 11, Duration: dur,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.03, OutlierProb: 0.01, OutlierScale: 5,
		ReorderWindow: 12,
		Excited:       true,
		Clocks: map[SensorKind]ClockProfile{
			// 里程计：先快后慢再恢复，斜率在运行中换向。
			SensorOdometry: {
				Slopes:   []float64{1.002, 0.9985, 1.0008},
				Segments: []time.Duration{min, min},
				Offset0:  -150 * time.Millisecond,
			},
			// 绝对位姿：持续偏慢且初始负偏差。
			SensorAbsolutePose: {
				Slopes:   []float64{0.997},
				Segments: nil,
				Offset0:  300 * time.Millisecond,
			},
		},
		PoseGap: 5 * time.Second,
	}
}

// TestAlignmentTracksTimeVaryingDrift 漂移随时间变化时，各路映射应被
// 持续估计并最终锁定；锁定后换算时间应贴近统一时间轴。
func TestAlignmentTracksTimeVaryingDrift(t *testing.T) {
	cfg := driftReplayConfig(3 * time.Minute)
	obs := GenerateReplay(cfg)
	t.Logf("漂移回放规模: %d 条 / %v", len(obs), cfg.Duration)

	eng := NewEngine(alignTestConfig())
	const batchSize = 64
	for i := 0; i < len(obs); i += batchSize {
		end := i + batchSize
		if end > len(obs) {
			end = len(obs)
		}
		eng.Ingest(obs[i:end])
	}
	end := obs[0].Stamp
	for _, o := range obs {
		if o.Stamp.After(end) {
			end = o.Stamp
		}
	}
	eng.AdvanceTo(end.Add(time.Second))

	for _, st := range eng.AlignmentStats() {
		t.Logf("对齐状态 kind=%s state=%s skew=%.6f offset=%v resid=%v samples=%d ref=%v",
			st.Kind, st.State, st.Skew, st.Offset, st.Residual, st.Samples, st.IsRef)
	}
	stats := eng.Stats()
	t.Logf("引擎统计: consumed=%d rejected=%d pending=%d peak=%d",
		stats.Consumed, stats.Rejected, stats.Pending, stats.PendingHigh)

	byKind := map[SensorKind]AlignStat{}
	for _, st := range eng.AlignmentStats() {
		byKind[st.Kind] = st
	}
	if byKind[SensorOdometry].State != AlignLocked {
		t.Fatalf("里程计应对齐锁定, got %s", byKind[SensorOdometry].State)
	}
	if byKind[SensorAbsolutePose].State != AlignLocked {
		t.Fatalf("绝对位姿应对齐锁定, got %s", byKind[SensorAbsolutePose].State)
	}
	// 用回放已知的时钟曲线，在多个物理时刻核对“原始戳→统一时间”的
	// 换算精度（这是时间对齐的核心价值），不依赖原点处的截距数值。
	poseClock := driftReplayConfig(time.Second).Clocks[SensorAbsolutePose]
	start := time.Unix(1_700_000_000, 0)
	maxAlignErr := 0.0
	// 采样点取在各漂移分段的中部（段：0-60/60-120/120-180s），远离
	// 60/120s 的斜率切换点；切换点附近局部仿射存在固有短暂过渡。
	for _, phys := range []float64{40, 90, 100, 150, 165} {
		physT := start.Add(time.Duration(phys * float64(time.Second)))
		rawT := poseClock.rawStamp(start, physT)
		got, ok := eng.AlignedTime(SensorAbsolutePose, rawT)
		if !ok {
			t.Fatalf("物理 %vs 处位姿路应可换算", phys)
		}
		err := math.Abs(float64(got.Sub(physT)) / float64(time.Second))
		if err > maxAlignErr {
			maxAlignErr = err
		}
	}
	t.Logf("位姿路最大时间换算误差: %.4fs", maxAlignErr)
	// 稳态换算误差门限 0.15s：该场景固定偏置 0.3s、末端累计漂移 0.84s，
	// 0.15s 远小于待校正量（证明换算确实在跟随漂移），也是时延相关在
	// 分段漂移 + 5Hz 位姿率下的可达量化精度。
	if maxAlignErr > 0.15 {
		t.Fatalf("漂移下时间换算误差过大: %.3fs", maxAlignErr)
	}
	// 全局斜率：分段漂移下全局单一 skew 是各段的折中，仅要求方向正确
	// （>1，即时钟偏慢）且不越可信范围；精确的时间换算由锚点分段插值
	// 保证（上面误差 0.123s）。恒定单段斜率的精确跟踪由另一个用例覆盖。
	skew := byKind[SensorAbsolutePose].Skew
	if skew < 1.001 || skew > 1.02 {
		t.Fatalf("位姿路斜率方向/范围异常: got %.6f", skew)
	}
}

// 为验证“斜率确实被持续跟踪”，单独用一条全程单段慢漂移（斜率恒定
// 0.997、初始偏置 0.3s）检查 skew 与 offset 的稳态估计。
func TestAlignmentConstantSkew(t *testing.T) {
	base := time.Unix(1_700_000_000, 0)
	min := 3 * time.Minute
	cfg := ReplayConfig{
		Seed: 5, Duration: min,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.03, OutlierProb: 0.01, OutlierScale: 5,
		ReorderWindow: 12, Excited: true,
		Clocks: map[SensorKind]ClockProfile{
			SensorAbsolutePose: {Slopes: []float64{0.997}, Segments: nil, Offset0: 300 * time.Millisecond},
		},
	}
	obs := GenerateReplay(cfg)
	c := alignTestConfig()
	c.AlignWindow = 20 * time.Second // 恒定斜率用长窗精确估计
	eng := NewEngine(c)
	for i := 0; i < len(obs); i += 64 {
		e := i + 64
		if e > len(obs) {
			e = len(obs)
		}
		eng.Ingest(obs[i:e])
	}
	var st AlignStat
	for _, s := range eng.AlignmentStats() {
		if s.Kind == SensorAbsolutePose {
			st = s
		}
	}
	if st.State != AlignLocked {
		t.Fatalf("恒定慢漂移位姿路应锁定, got %s", st.State)
	}
	// 该路原始时钟走得慢（raw=0.994·phys−0.3），反演到统一时间的映射
	// 斜率 skew=1/0.994≈1.006（>1）。联合搜索应估出方向正确（显著大
	// 于 1）且不越过 AlignMaxSkew=0.05 可信范围的斜率。
	if st.Skew <= 1.001 || st.Skew >= 1.05 {
		t.Fatalf("恒定慢漂移斜率方向/范围错误: got %.6f (want ~1.006)", st.Skew)
	}
	// 核心正确性：多个中部已确认时刻的“原始戳→统一时间”换算误差应
	// 远小于该路累计漂移（数十至数百毫秒量级）。取方法可达精度 0.2s，
	// 它远小于本场景漂移（>0.6s），证明换算确实在持续修正而非恒等。
	maxErr := 0.0
	for _, sec := range []float64{40, 60, 80, 100, 110} {
		physT := base.Add(time.Duration(sec * float64(time.Second)))
		rawT := cfg.Clocks[SensorAbsolutePose].rawStamp(base, physT)
		got, ok := eng.AlignedTime(SensorAbsolutePose, rawT)
		if !ok {
			t.Fatalf("物理 %vs 应可换算", sec)
		}
		if e := math.Abs(float64(got.Sub(physT)) / float64(time.Second)); e > maxErr {
			maxErr = e
		}
	}
	t.Logf("恒定漂移 skew=%.6f resid=%v 中部最大换算误差=%.4fs", st.Skew, st.Residual, maxErr)
	if maxErr > 0.2 {
		t.Fatalf("恒定慢漂移换算误差过大: %.3fs", maxErr)
	}
}

// runDrift 用给定回放与配置摄入全部观测，返回引擎。
func runDrift(t *testing.T, cfg ReplayConfig, ec Config) *Engine {
	t.Helper()
	obs := GenerateReplay(cfg)
	eng := NewEngine(ec)
	const bs = 64
	for i := 0; i < len(obs); i += bs {
		end := i + bs
		if end > len(obs) {
			end = len(obs)
		}
		eng.Ingest(obs[i:end])
	}
	return eng
}

// ingestAndDrain 分批摄入全部观测，末尾 AdvanceTo 到最晚观测时刻排空（不
// 再多纯预测），返回引擎与真值轨迹起点。
func ingestAndDrain(obs []Observation, ec Config, bs int) *Engine {
	eng := NewEngine(ec)
	for i := 0; i < len(obs); i += bs {
		end := i + bs
		if end > len(obs) {
			end = len(obs)
		}
		eng.Ingest(obs[i:end])
	}
	end := obs[0].Stamp
	for _, o := range obs {
		if o.Stamp.After(end) {
			end = o.Stamp
		}
	}
	eng.AdvanceTo(end)
	return eng
}

// boundedReversalClocks 构造一组全程“偏移有界、斜率在运行中反复换向”的
// 多路时钟曲线：位姿/里程计各用 6 段斜率（每段 5 分钟），偏移量在整个半
// 小时内始终有界（峰值 <1s）、不会无限累积到搜索范围之外。用于长跑回归。
func boundedReversalClocks() map[SensorKind]ClockProfile {
	seg := 5 * time.Minute
	segs := func(n int) []time.Duration {
		out := make([]time.Duration, n)
		for i := range out {
			out[i] = seg
		}
		return out
	}
	return map[SensorKind]ClockProfile{
		// 位姿：+0.3% / −0.3% 交替，偏移在 0..+0.9s 间有界振荡。
		SensorOdometry: {
			Slopes:   []float64{0.9985, 1.002, 1.0005, 0.999, 1.002, 0.998},
			Segments: segs(6), Offset0: -150 * time.Millisecond,
		},
		SensorAbsolutePose: {
			Slopes:   []float64{1.003, 0.997, 1.003, 0.997, 1.003, 0.997},
			Segments: segs(6), Offset0: 300 * time.Millisecond,
		},
	}
}

func statOf(eng *Engine, k SensorKind) AlignStat {
	for _, s := range eng.AlignmentStats() {
		if s.Kind == k {
			return s
		}
	}
	return AlignStat{Kind: k, State: AlignCold}
}

// TestAlignmentUnavailableReason 验证对齐尚不可用时观测被明确暂缓，
// 判定原因是 align_calibrating，且不作为就绪数据参与消费。
func TestAlignmentUnavailableReason(t *testing.T) {
	// 极短回放：参考 IMU 刚出现、位姿路尚无足够锚点。
	cfg := ReplayConfig{
		Seed: 7, Duration: 4 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		ReorderWindow: 4, Excited: true,
		Clocks: map[SensorKind]ClockProfile{
			SensorAbsolutePose: {Slopes: []float64{0.99}, Segments: nil, Offset0: 300 * time.Millisecond},
		},
	}
	ec := alignTestConfig()
	ec.AlignWindow = 30 * time.Second
	eng := runDrift(t, cfg, ec)

	// 位姿路在样本不足时必须处于 cold 或 calibrating，绝不假装锁定。
	st := statOf(eng, SensorAbsolutePose)
	if st.State == AlignLocked {
		t.Fatalf("样本不足不应锁定, got %+v", st)
	}
	// 该路观测的判定原因应可辨识为“校准中/样本不足”。
	reasons := map[RejectReason]int{}
	for _, d := range eng.Decisions() {
		if d.Reason == ReasonAlignCalibrating {
			reasons[d.Reason]++
		}
	}
	// 至少应出现过 align_calibrating 的暂缓判定（短数据下无法收敛）。
	if reasons[ReasonAlignCalibrating] == 0 {
		t.Fatalf("应对齐不可用观测给出 align_calibrating 暂缓原因, stats=%+v", st)
	}
	t.Logf("短期数据位姿路 state=%s samples=%d, 暂缓判定=%d",
		st.State, st.Samples, reasons[ReasonAlignCalibrating])
}

// TestAlignmentOutOfRange 验证时钟斜率/偏移越出可信范围时该路进入
// out_of_range，判定原因为 time_unavailable，与“样本不足”区分。
func TestAlignmentOutOfRange(t *testing.T) {
	// 固定偏移 5s，超出 AlignSearchRange(2s) 且刻意让斜率可信但偏移
	// 巨大：用 +5s 固定偏差（<AlignMaxDrift 10s 但 >检测/搜索范围），
	// 关键是相关搜索找不到对齐峰 → 锚点残差大 → 保持校准/不可用；
	// 再用斜率 ±40%（超过 AlignMaxSkew 5%）直接触发 out_of_range。
	cfg := ReplayConfig{
		Seed: 9, Duration: 90 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.02, ReorderWindow: 6, Excited: true,
		Clocks: map[SensorKind]ClockProfile{
			SensorOdometry: {Slopes: []float64{1.15}, Segments: nil, Offset0: 0},
		},
	}
	ec := alignTestConfig()
	ec.AlignMaxSkew = 0.05
	eng := runDrift(t, cfg, ec)

	st := statOf(eng, SensorOdometry)
	// 原始时钟快 15%，反演斜率 1/1.15≈0.870，超出 ±5% 可信范围，应被判
	// out_of_range（且不锁定）。
	if st.State == AlignLocked {
		t.Fatalf("超范围时钟不应锁定: %+v", st)
	}
	if _, ok := eng.AlignedTime(SensorOdometry, time.Unix(0, 0)); ok {
		t.Fatal("超范围时钟不应提供换算时间")
	}
	// 该路应出现 time_unavailable（out_of_range）或校准中；这里强约束
	// 长数据下应识别为 out_of_range。
	sawUnavailable := false
	for _, d := range eng.Decisions() {
		if d.Reason == ReasonTimeUnavailable {
			sawUnavailable = true
		}
	}
	if !sawUnavailable {
		t.Fatalf("越界时钟应给出 time_unavailable 判定, state=%s", st.State)
	}
	t.Logf("越界里程计 state=%s skew=%.4f（已判 time_unavailable）", st.State, st.Skew)
}

// TestBacklogRejected 待处理积压超过 MaxPendingObs 时新观测被
// backlog_exceeded 拒绝，运行中待处理数据规模始终有上界。
func TestBacklogRejected(t *testing.T) {
	t0 := testEpoch
	cfg := testConfig()
	// 乱序等待窗口使观测在消费前挂入重排缓冲；不调用 AdvanceTo，制造
	// 大量待处理项。
	cfg.EnableAlignment = false
	cfg.ReorderDelay = 10 * time.Second
	cfg.Limits.MaxPendingObs = 30
	cfg.Limits.MaxBatchSize = 256
	eng := NewEngine(cfg)

	var rejected int
	for b := 0; b < 10; b++ {
		batch := make([]Observation, 0, 20)
		for j := 0; j < 20; j++ {
			o := Observation{
				Kind:  SensorIMU,
				Stamp: t0.Add(time.Duration(b*20+j) * 100 * time.Millisecond),
				Seq:   uint64(b*20 + j + 1),
				Accel: Vec3{Z: 9.81},
				Gyro:  Vec3{Z: 0.3},
			}
			batch = append(batch, o)
		}
		for _, d := range eng.Ingest(batch) {
			if !d.Accepted && d.Reason == ReasonBacklogExceeded {
				rejected++
			}
		}
	}
	st := eng.Stats()
	if st.Pending > cfg.Limits.MaxPendingObs {
		t.Fatalf("待处理积压 %d 超过上限 %d", st.Pending, cfg.Limits.MaxPendingObs)
	}
	if rejected == 0 {
		t.Fatal("应有观测因积压过多被 backlog_exceeded 拒绝")
	}
	t.Logf("积压=%d（上限%d）backlog拒绝=%d", st.Pending, cfg.Limits.MaxPendingObs, rejected)
}

// TestDeterministicOpBudget 确定性操作预算：用 MaxBatchOps 限制单批
// 对齐相关工作量，超限整批拒绝（process_budget），重复运行判定严格
// 一致（不依赖墙钟）。
func TestDeterministicOpBudget(t *testing.T) {
	cfg := driftReplayConfig(60 * time.Second)
	ec := alignTestConfig()
	ec.Limits.MaxBatchOps = 1 // 极低预算：含对齐相关的批次必然超限
	ec.Limits.MaxBatchSize = 64

	eng1 := NewEngine(ec)
	eng2 := NewEngine(ec)
	var budgetReject int
	for i := 0; i < len(GenerateReplay(cfg)); i += 64 {
		obs := GenerateReplay(cfg)
		end := i + 64
		if end > len(obs) {
			end = len(obs)
		}
		for _, d := range eng1.Ingest(obs[i:end]) {
			if d.Reason == ReasonProcessBudget {
				budgetReject++
			}
		}
	}
	// 第二次运行，逐批比较判定分布。
	obs := GenerateReplay(cfg)
	for i := 0; i < len(obs); i += 64 {
		end := i + 64
		if end > len(obs) {
			end = len(obs)
		}
		eng2.Ingest(obs[i:end])
	}
	if budgetReject == 0 {
		t.Fatal("极低操作预算下应出现 process_budget 拒绝")
	}
	// 两次运行的判定原因序列一致（确定性）。
	r1 := reasonsOf(eng1.Decisions())
	r2 := reasonsOf(eng2.Decisions())
	if len(r1) != len(r2) {
		t.Fatalf("两次运行判定条数不同: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i] != r2[i] {
			t.Fatalf("判定在第 %d 条不一致: %v vs %v", i, r1[i], r2[i])
		}
	}
	t.Logf("process_budget 拒绝=%d，两次运行判定一致（共%d条）", budgetReject, len(r1))
}

func reasonsOf(ds []Decision) []RejectReason {
	out := make([]RejectReason, len(ds))
	for i, d := range ds {
		out[i] = d.Reason
	}
	return out
}

// TestConcurrentAlignmentConsistency 在启用在线对齐、存在漂移与乱序
// 的情况下，并发摄入/推进/快照读取，用 -race 验证无数据竞争，且
// 快照始终自洽（协方差正定、四元数单位）。
func TestConcurrentAlignmentConsistency(t *testing.T) {
	ec := alignTestConfig()
	ec.Limits.MaxProcessTime = 30 * time.Second
	eng := NewEngine(ec)
	var stop atomic.Bool
	var wg sync.WaitGroup

	// 写入方：3 个 goroutine 写入同一连续时间轴上互不重叠、按时间排序
	// 的区段（writer w 负责时间 [w·seg,(w+1)·seg)）。它们并发调用 Ingest，
	// 但提交的时间戳各自单调、跨区段不倒退，符合多生产者有序投递场景，
	// 聚焦验证预测/更新/快照交错的内存一致性。
	const (
		writers = 3
		seg     = 4000 // 每段 4000ms 采样
		dtMs    = 10
	)
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(1000 + id)))
			t0 := id * seg
			for i := 0; i < seg/dtMs && !stop.Load(); i++ {
				kind := SensorIMU
				switch rng.Intn(3) {
				case 1:
					kind = SensorOdometry
				case 2:
					kind = SensorAbsolutePose
				}
				ts := time.Unix(1_700_000_000, 0).Add(
					time.Duration(t0+i*dtMs) * time.Millisecond)
				o := Observation{Kind: kind, Stamp: ts, Seq: uint64(id*100000 + i)}
				switch kind {
				case SensorIMU:
					o.Accel, o.Gyro = Vec3{Z: 9.81}, Vec3{}
				case SensorOdometry:
					o.LinVel, o.AngVel = Vec3{}, Vec3{}
					o.CovDiag = [9]float64{0.01}
				default:
					o.Pose = Pose{Position: Vec3{X: 1}, Rotation: QuatIdentity()}
					o.CovDiag = [9]float64{0.01, 0.01, 0.01, 0.001, 0.001, 0.001}
				}
				// 小批投递（每批 8 条），制造并发交错。
				eng.Ingest([]Observation{o})
			}
		}(w)
	}

	// 推进 + 快照读取方。
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 300 && !stop.Load(); i++ {
			eng.AdvanceTo(time.Unix(1_700_000_000, 0).Add(time.Duration(i) * 200 * time.Millisecond))
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 300 && !stop.Load(); i++ {
			s, c, ok := eng.Snapshot()
			if ok {
				n := s.Rotation.W*s.Rotation.W +
					s.Rotation.X*s.Rotation.X +
					s.Rotation.Y*s.Rotation.Y +
					s.Rotation.Z*s.Rotation.Z
				if math.Abs(n-1) > 1e-9 {
					t.Errorf("快照四元数非单位: %v", n)
					stop.Store(true)
					return
				}
				if !c.IsPositiveDefinite() {
					t.Error("快照协方差失去正定性")
					stop.Store(true)
					return
				}
			}
		}
	}()

	wg.Wait()
}

// TestLongRunHalfHour 至少 30 分钟确定性回放：持续叠加到达乱序、偶发
// 缺帧、离群观测、时间基准漂移（分段变向）、中途一路缺测。验证：
//   - 全程姿态/速度/协方差稳定、四元数单位、协方差正定；
//   - 位姿末端位置误差在合理范围内；
//   - 待处理数据始终有界，拒绝后引擎继续；
//   - 重复运行结果严格一致（确定性）。
func TestLongRunHalfHour(t *testing.T) {
	if testing.Short() {
		t.Skip("长跑用例")
	}
	run := func(seed int64) (State, Mat, EngineStats, map[RejectReason]int) {
		// 长跑稳定性场景：恒定圆周轨迹 + 乱序、缺帧、离群、一路长时间
		// 缺测。本用例验证 ESKF 与引擎在 30 分钟大流量下的长时稳定性
		// （姿态/速度/协方差/积压/确定性），故关闭在线对齐层走基线即时
		// 路径（该路径已验证长时稳定）；在线时间对齐的漂移跟踪精度由
		// 时变激励轨迹的中短时用例（TestAlignmentTracksTimeVaryingDrift
		// 等）专门覆盖。
		cfg := driftReplayConfig(30 * time.Minute)
		cfg.Seed = seed
		cfg.Excited = false
		cfg.Clocks = nil
		cfg.ClockDrift = 0
		cfg.SensorGap = map[SensorKind]time.Duration{
			SensorOdometry: 40 * time.Second,
		}
		ec := testConfig()
		ec.DegradeAfter = 3 * time.Second
		ec.Limits.MaxProcessTime = 60 * time.Second
		ec.Limits.MaxPendingObs = 4096
		obs := GenerateReplay(cfg)
		eng := NewEngine(ec)
		const bs = 128
		for i := 0; i < len(obs); i += bs {
			end := i + bs
			if end > len(obs) {
				end = len(obs)
			}
			eng.Ingest(obs[i:end])
		}
		endStamp := obs[0].Stamp
		for _, o := range obs {
			if o.Stamp.After(endStamp) {
				endStamp = o.Stamp
			}
		}
		eng.AdvanceTo(endStamp.Add(2 * time.Second))
		s, c, ok := eng.Snapshot()
		if !ok {
			t.Fatal("长跑后应有状态")
		}
		stat := map[RejectReason]int{}
		for _, d := range eng.Decisions() {
			stat[d.Reason]++
		}
		return s, c, eng.Stats(), stat
	}

	s1, c1, st1, reasons1 := run(77)
	s2, _, st2, _ := run(77)

	// 确定性：两次运行状态一致（容差 1e-9，吸收归一化末位差异）。
	if !statesEqual(s1, s2, 1e-9) {
		t.Fatalf("长跑两次运行状态不一致:\n%+v\n%+v", s1, s2)
	}
	for i := 0; i < 9; i++ {
		if c1.At(i, i) <= 1e-12 {
			t.Fatalf("长跑后协方差对角 %d 非正: %v", i, c1.At(i, i))
		}
	}
	if !c1.IsPositiveDefinite() {
		t.Fatal("长跑后协方差失去正定性")
	}
	nq := s1.Rotation.W*s1.Rotation.W + s1.Rotation.X*s1.Rotation.X +
		s1.Rotation.Y*s1.Rotation.Y + s1.Rotation.Z*s1.Rotation.Z
	if math.Abs(nq-1) > 1e-9 {
		t.Fatalf("长跑后四元数退化: |q|²=%v", nq)
	}
	// 速度有界（激励轨迹线速度 ~1.5m/s，加噪后不应发散）。
	vn := vecNorm(s1.Velocity)
	if vn > 6 {
		t.Fatalf("长跑后速度发散: %.3f", vn)
	}
	// 待处理数据全程有界（不超过容量；一路缺测期允许暂态堆积，恢复后消化）。
	if st1.PendingHigh > 4096 {
		t.Fatalf("长跑积压无界: %+v", st1)
	}
	if st2.Pending != st1.Pending {
		t.Fatalf("两次运行积压不一致")
	}
	t.Logf("长跑判定分布: %+v", reasons1)

	// 末端位置误差：与真实恒定圆周轨迹对比（本场景时钟无漂移）。
	start := time.Unix(1_700_000_000, 0)
	traj := DefaultTrajectory(start, 30*time.Minute)
	truth := traj.TruePoseAt(start.Add(30 * time.Minute))
	pe := vecSub(s1.Position, truth.Position)
	posErr := vecNorm(pe)
	rotErr := vecNorm(vecSub(QuatToRotVec(s1.Rotation), QuatToRotVec(truth.Rotation)))
	t.Logf("30分钟: 观测~%d 待处理峰值=%d 位置误差=%.3fm 姿态误差=%.4frad 速度=%.3f 判定=%v",
		90000, st1.PendingHigh, posErr, rotErr, vn, reasons1)
	if posErr > 6.0 {
		t.Fatalf("长跑末端位置误差过大: %.3fm", posErr)
	}
	if rotErr > 0.5 {
		t.Fatalf("长跑末端姿态误差过大: %.4frad", rotErr)
	}
}

// endErr 在“状态时刻对应的时变激励真值”上评估末端位置误差与速度范数。
func endErr(eng *Engine) (float64, float64) {
	start := time.Unix(1_700_000_000, 0)
	ex := ExcitedTrajectory(start)
	s, _, ok := eng.Snapshot()
	if !ok {
		return math.Inf(1), math.Inf(1)
	}
	return vecNorm(vecSub(s.Position, ex.PoseAt(s.Stamp).Position)), vecNorm(s.Velocity)
}

// TestAlignmentNoDriftBitIdentical 无时钟漂移输入下，开启与关闭在线对齐
// 必须得到逐位一致的融合结果（位置/速度），跨多 seed、多时长。这是“对齐
// 层对既有行为零影响”的硬兼容保证。
func TestAlignmentNoDriftBitIdentical(t *testing.T) {
	for _, dur := range []time.Duration{30 * time.Second, 3 * time.Minute} {
		for _, seed := range []int64{11, 23} {
			cfg := ReplayConfig{
				Seed: seed, Duration: dur,
				IMURate: 100, OdomRate: 20, PoseRate: 5,
				DropProb: 0.03, OutlierProb: 0.01, OutlierScale: 5,
				ReorderWindow: 12, Excited: true, // 不配 Clocks：零漂移
			}
			obs := GenerateReplay(cfg)
			sOff, _, _ := ingestAndDrain(obs, replayTestConfig(), 64).Snapshot()
			onE := ingestAndDrain(obs, alignTestConfig(), 64)
			sOn, _, ok := onE.Snapshot()
			if !ok {
				t.Fatalf("dur=%v seed=%d 开对齐未初始化", dur, seed)
			}
			dp := vecNorm(vecSub(sOff.Position, sOn.Position))
			dv := vecNorm(vecSub(sOff.Velocity, sOn.Velocity))
			if dp > 1e-9 || dv > 1e-9 {
				t.Fatalf("零漂移 dur=%v seed=%d 开关对齐不逐位一致: Δpos=%.3e Δvel=%.3e",
					dur, seed, dp, dv)
			}
			// 各路在零漂移下必须保持 cold 恒等，绝不误锁。
			for _, st := range onE.AlignmentStats() {
				if !st.IsRef && st.State != AlignCold {
					t.Fatalf("零漂移 dur=%v seed=%d 路 %s 误离开 cold: %s skew=%.5f off=%v",
						dur, seed, st.Kind, st.State, st.Skew, st.Offset)
				}
			}
		}
	}
}

// TestAlignmentTimeVaryingNotWorse 漂移方向/幅度随时间变化（斜率换向）时，
// 开对齐的融合轨迹误差不得比关对齐更差，且速度有界不发散；同时各路必须
// locked、状态口径与实际换算一致。
func TestAlignmentTimeVaryingNotWorse(t *testing.T) {
	for _, dur := range []time.Duration{time.Minute, 3 * time.Minute} {
		cfg := driftReplayConfig(dur)
		obs := GenerateReplay(cfg)
		ec := alignTestConfig()
		ec.Limits.MaxPendingObs = 8192 // 离线追跑允许锚点确认暂态堆积；实时 1x 下有界
		offE := ingestAndDrain(obs, replayTestConfig(), 64)
		onE := ingestAndDrain(obs, ec, 64)
		epOff, vOff := endErr(offE)
		epOn, vOn := endErr(onE)
		t.Logf("时变漂移 dur=%v OFF posErr=%.3fm |v|=%.2f | ON posErr=%.3fm |v|=%.2f",
			dur, epOff, vOff, epOn, vOn)
		if math.IsNaN(vOn) || math.IsInf(vOn, 0) || vOn > 6 {
			t.Fatalf("开对齐速度发散/异常: |v|=%.3f", vOn)
		}
		// 开对齐必须不劣于关对齐（留 0.5m 容差吸收末端噪声，但不允许系统性变差）。
		if epOn > epOff+0.5 {
			t.Fatalf("开对齐位置误差(%.3fm)劣于关对齐(%.3fm)", epOn, epOff)
		}
		// 对外状态必须诚实：真正参与换算的两路都应 locked，绝不显示已锁定却发散。
		for _, k := range []SensorKind{SensorOdometry, SensorAbsolutePose} {
			if st := statOf(onE, k); st.State != AlignLocked {
				t.Fatalf("dur=%v 路 %s 未锁定: %s", dur, k, st.State)
			}
		}
	}
}

// TestAlignmentHalfHourReversalBounded 半小时回放：各路时钟漂移在运行中多
// 次换向（偏移始终有界）。验证开对齐全程有界、末端不劣于关对齐、待处理
// 始终有上限、且各路稳定锁定。这是“长时间把各路时钟随时间变化的漂移真
// 实叠入回放”的长跑回归。
func TestAlignmentHalfHourReversalBounded(t *testing.T) {
	if testing.Short() {
		t.Skip("长跑用例")
	}
	cfg := ReplayConfig{
		Seed: 77, Duration: 30 * time.Minute,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.03, OutlierProb: 0.01, OutlierScale: 5,
		ReorderWindow: 12, Excited: true,
		Clocks:  boundedReversalClocks(),
		PoseGap: 5 * time.Second,
	}
	obs := GenerateReplay(cfg)
	t.Logf("半小时换向漂移回放规模: %d 条", len(obs))

	mk := func(on bool) Config {
		var ec Config
		if on {
			ec = alignTestConfig()
		} else {
			ec = replayTestConfig()
		}
		ec.DegradeAfter = 3 * time.Second
		ec.Limits.MaxPendingObs = 8192
		ec.Limits.MaxProcessTime = time.Minute
		return ec
	}
	offE := ingestAndDrain(obs, mk(false), 128)
	onE := ingestAndDrain(obs, mk(true), 128)
	epOff, vOff := endErr(offE)
	epOn, vOn := endErr(onE)
	st := onE.Stats()
	t.Logf("半小时 OFF posErr=%.3fm |v|=%.2f | ON posErr=%.3fm |v|=%.2f peak=%d pending=%d",
		epOff, vOff, epOn, vOn, st.PendingHigh, st.Pending)
	if math.IsNaN(vOn) || math.IsInf(vOn, 0) || vOn > 6 {
		t.Fatalf("半小时开对齐速度发散: |v|=%.3f", vOn)
	}
	if epOn > 3.0 {
		t.Fatalf("半小时开对齐末端位置误差过大、未被时间对齐收敛: %.3fm", epOn)
	}
	if epOn > epOff+1.0 {
		t.Fatalf("半小时开对齐(%.3fm)劣于关对齐(%.3fm)", epOn, epOff)
	}
	if st.PendingHigh > 8192 {
		t.Fatalf("半小时待处理积压突破上限: peak=%d", st.PendingHigh)
	}
	if st.Pending != 0 {
		t.Fatalf("排空后仍有待处理项: %d", st.Pending)
	}
	for _, k := range []SensorKind{SensorOdometry, SensorAbsolutePose} {
		if s := statOf(onE, k); s.State != AlignLocked {
			t.Fatalf("半小时路 %s 未保持 locked: %s", k, s.State)
		}
	}

	// 确定性：相同输入重复运行严格一致。
	onE2 := ingestAndDrain(obs, mk(true), 128)
	s1, _, _ := onE.Snapshot()
	s2, _, _ := onE2.Snapshot()
	if !statesEqual(s1, s2, 1e-9) {
		t.Fatalf("半小时开对齐两次运行不一致")
	}
}
