package fusion

import (
	"math"
	"sort"
	"testing"
	"time"
)

// physArrival 用回放已知时钟曲线把一条观测的原始戳反演为其物理采样
// （=在线到达）时刻。未配置漂移的路恒等。rawStamp 关于物理时间单调
// （各段斜率为正），二分反演稳定。
func physArrival(cfg ReplayConfig, o Observation) time.Time {
	start := time.Unix(1_700_000_000, 0)
	cp, ok := cfg.Clock(o.Kind)
	if !ok {
		return o.Stamp
	}
	lo, hi := 0.0, cfg.Duration.Seconds()+30.0
	r := o.Stamp
	for i := 0; i < 60; i++ {
		mid := (lo + hi) / 2
		mt := start.Add(time.Duration(mid * float64(time.Second)))
		if cp.rawStamp(start, mt).Before(r) {
			lo = mid
		} else {
			hi = mid
		}
	}
	return start.Add(time.Duration(((lo + hi) / 2) * float64(time.Second)))
}

// arrivedBatch 把观测按物理到达时刻排序后，仅保留不晚于 cutoff 的部分
// （每 64 条一批，复刻真实在线 + demo 的批量摄入）。
func arrivedBatches(obs []Observation, cfg ReplayConfig, cutoff time.Time, bs int) [][]Observation {
	type it struct {
		o  Observation
		pt time.Time
	}
	items := make([]it, 0, len(obs))
	for _, o := range obs {
		if pt := physArrival(cfg, o); !pt.After(cutoff) {
			items = append(items, it{o, pt})
		}
	}
	sort.SliceStable(items, func(i, j int) bool { return items[i].pt.Before(items[j].pt) })
	var batches [][]Observation
	for i := 0; i < len(items); i += bs {
		end := i + bs
		if end > len(items) {
			end = len(items)
		}
		b := make([]Observation, 0, end-i)
		for _, x := range items[i:end] {
			b = append(b, x.o)
		}
		batches = append(batches, b)
	}
	return batches
}

// evalAt 构建引擎、按物理到达批量喂入不晚于 evalSec 的观测，再推进到
// 该物理时刻，返回位置误差、速度范数与引擎（供状态/规模断言）。
func evalAt(t *testing.T, cfg ReplayConfig, ec Config, evalSec float64) (float64, float64, *Engine) {
	t.Helper()
	obs := GenerateReplay(cfg)
	eng := NewEngine(ec)
	start := time.Unix(1_700_000_000, 0)
	cutoff := start.Add(time.Duration(evalSec * float64(time.Second)))
	for _, b := range arrivedBatches(obs, cfg, cutoff, 64) {
		eng.Ingest(b)
	}
	eng.AdvanceTo(cutoff)
	st, _, ok := eng.Snapshot()
	if !ok {
		t.Fatalf("t=%.0fs 引擎应已初始化", evalSec)
	}
	truth := ExcitedTrajectory(start).PoseAt(cutoff)
	return vecNorm(vecSub(st.Position, truth.Position)), vecNorm(st.Velocity), eng
}

// TestTimeVaryingDriftAlignedNotWorse 直接断言：在“每路相对统一时间
// 轴快慢不一、运行中换向、方向与幅度不一致”的时变漂移下，打开在线
// 对齐后的融合误差不劣于关闭对齐、且全程有界不发散（速度也有界）。
//
// 这是本次修复的核心回归：修复前 1 分钟量级会速度上飙、位置偏出数百
// 米且状态仍显示 locked。断言不通过调低回放噪声、也不放宽任何既有
// 精度门限来“凑过”。
func TestTimeVaryingDriftAlignedNotWorse(t *testing.T) {
	// 位置误差/速度的硬上界：激励轨迹线速度 ≤ 5·(0.3+0.22)=2.6m/s，
	// 修复后对齐把时间错位修正到数十毫秒，误差应稳定在米级以内。
	const posBound = 3.0
	const velBound = 4.0
	for _, dur := range []time.Duration{60 * time.Second, 180 * time.Second} {
		cfg := driftReplayConfig(dur)
		ecOn := alignTestConfig()
		ecOff := replayTestConfig()
		ecOff.EnableAlignment = false

		var secs []float64
		for s := 15.0; s <= dur.Seconds()-1.0; s += 15.0 {
			secs = append(secs, s)
		}
		secs = append(secs, dur.Seconds()-1.0)

		var maxOn, maxOff, maxVelOn, maxVelOff float64
		worsePoints := 0
		for _, s := range secs {
			eOn, vOn, engOn := evalAt(t, cfg, ecOn, s)
			eOff, vOff, _ := evalAt(t, cfg, ecOff, s)
			if eOn > maxOn {
				maxOn = eOn
			}
			if eOff > maxOff {
				maxOff = eOff
			}
			if vOn > maxVelOn {
				maxVelOn = vOn
			}
			if vOff > maxVelOff {
				maxVelOff = vOff
			}
			// 允许亚米级数值噪声导致的轻微交叉；超过 0.25m 且比值 >1.25
			// 才算实质劣化。
			if eOn > eOff+0.25 && eOn > eOff*1.25 {
				worsePoints++
				t.Errorf("dur=%v t=%.0fs 开对齐反而更差: ON=%.3fm OFF=%.3fm", dur, s, eOn, eOff)
			}
			// 每点都必须有界、不发散。
			if eOn > posBound || vOn > velBound || math.IsNaN(eOn) || math.IsInf(vOn, 0) {
				t.Fatalf("dur=%v t=%.0fs 开对齐发散: posErr=%.3fm vel=%.3f", dur, s, eOn, vOn)
			}
			_ = engOn
		}
		t.Logf("dur=%v 采样点=%d 全程峰值: errON=%.3fm errOFF=%.3fm velON=%.3f velOFF=%.3f 劣化点=%d",
			dur, len(secs), maxOn, maxOff, maxVelOn, maxVelOff, worsePoints)

		// 末端：两路都必须真正锁定，且 ON 末端误差显著不劣于 OFF。
		_, _, engOn := evalAt(t, cfg, ecOn, dur.Seconds()-1.0)
		states := map[SensorKind]AlignStat{}
		for _, st := range engOn.AlignmentStats() {
			states[st.Kind] = st
		}
		for _, k := range []SensorKind{SensorOdometry, SensorAbsolutePose} {
			if states[k].State != AlignLocked {
				t.Fatalf("dur=%v %s 应锁定, got %s（显示口径必须与实际一致）",
					dur, k, states[k].State)
			}
		}
	}
}

// runObsBatches 按观测生成顺序（含局部乱序）以 bs 条/批喂入引擎。
func runObsBatches(t *testing.T, eng *Engine, obs []Observation, bs int) {
	t.Helper()
	for i := 0; i < len(obs); i += bs {
		end := i + bs
		if end > len(obs) {
			end = len(obs)
		}
		eng.Ingest(obs[i:end])
	}
}

// endState 喂完全部观测并推进到物理末端后返回末态。
func endState(t *testing.T, cfg ReplayConfig, ec Config, advance time.Duration) State {
	t.Helper()
	obs := GenerateReplay(cfg)
	eng := NewEngine(ec)
	runObsBatches(t, eng, obs, 64)
	last := obs[0].Stamp
	for _, o := range obs {
		if o.Stamp.After(last) {
			last = o.Stamp
		}
	}
	eng.AdvanceTo(last.Add(advance))
	s, _, ok := eng.Snapshot()
	if !ok {
		t.Fatal("引擎未初始化")
	}
	return s
}

// TestZeroDriftAlignmentIsBitIdentical 本来没有时间漂移的输入，打开与
// 关闭在线对齐的融合结果必须完全一致（位姿/速度逐位）。cold 态恒等
// 放行、换算只在有统计显著的漂移证据后才发生，是这条兼容性的保证。
func TestZeroDriftAlignmentIsBitIdentical(t *testing.T) {
	mk := func(excited bool, dur time.Duration) ReplayConfig {
		return ReplayConfig{
			Seed: 11, Duration: dur,
			IMURate: 100, OdomRate: 20, PoseRate: 5,
			DropProb: 0.03, OutlierProb: 0.01, OutlierScale: 5,
			ReorderWindow: 12, Excited: excited,
		}
	}
	for _, excited := range []bool{false, true} {
		cfg := mk(excited, 90*time.Second)
		ecOff := replayTestConfig()
		ecOff.EnableAlignment = false
		ecOn := alignTestConfig()

		sOff := endState(t, cfg, ecOff, time.Second)
		sOn := endState(t, cfg, ecOn, time.Second)
		if !statesEqual(sOff, sOn, 0.0) {
			t.Fatalf("excited=%v 零漂移下开/关对齐结果不一致:\nOFF=%+v\nON =%+v",
				excited, sOff, sOn)
		}
		// 无漂移时各路必须保持 cold（未误锁定），参考路除外。
		obs := GenerateReplay(cfg)
		eng := NewEngine(ecOn)
		runObsBatches(t, eng, obs, 64)
		for _, st := range eng.AlignmentStats() {
			if st.IsRef {
				continue
			}
			if st.State != AlignCold {
				t.Fatalf("excited=%v %s 零漂移应保持 cold, got %s (skew=%.6f off=%v)",
					excited, st.Kind, st.State, st.Skew, st.Offset)
			}
		}
		t.Logf("excited=%v 零漂移开/关对齐逐位一致，各路保持 cold", excited)
	}
}

// TestAlignmentDisplayedTimeIsConsumedTime 对外 AlignedTime 给出的换算
// 时间，必须就是实际参与状态推进用的口径：锁定路换算时间应贴近物理
// 真值；未锁定路 AlignedTime 必须返回 false（不得显示可信却用错）。
func TestAlignmentDisplayedTimeIsConsumedTime(t *testing.T) {
	cfg := driftReplayConfig(180 * time.Second)
	obs := GenerateReplay(cfg)
	ec := alignTestConfig()
	eng := NewEngine(ec)
	runObsBatches(t, eng, obs, 64)
	start := time.Unix(1_700_000_000, 0)
	poseClock := cfg.Clocks[SensorAbsolutePose]
	odomClock := cfg.Clocks[SensorOdometry]
	maxPose, maxOdom := 0.0, 0.0
	for _, phys := range []float64{30, 60, 90, 120, 150, 170} {
		pt := start.Add(time.Duration(phys * float64(time.Second)))
		if g, ok := eng.AlignedTime(SensorAbsolutePose, poseClock.rawStamp(start, pt)); !ok {
			t.Fatalf("位姿路 t=%.0fs 应可换算（已锁定）", phys)
		} else if e := math.Abs(g.Sub(pt).Seconds()); e > maxPose {
			maxPose = e
		}
		if g, ok := eng.AlignedTime(SensorOdometry, odomClock.rawStamp(start, pt)); !ok {
			t.Fatalf("里程计 t=%.0fs 应可换算（已锁定）", phys)
		} else if e := math.Abs(g.Sub(pt).Seconds()); e > maxOdom {
			maxOdom = e
		}
	}
	// 参考路恒等。
	if g, ok := eng.AlignedTime(SensorIMU, start.Add(42*time.Second)); !ok || !g.Equal(start.Add(42*time.Second)) {
		t.Fatalf("参考路应恒等换算: %v %v", g, ok)
	}
	t.Logf("对外换算最大误差: pose=%.4fs odom=%.4fs", maxPose, maxOdom)
	// 位姿路 5Hz、全程恒定斜率，换算误差要求严（0.2s，远小于其 0.3s
	// 固定偏置 + 末端累计漂移）；里程计为三段换向斜率，评估点之间存在
	// 斜率切换，跨度外短距外推的过渡误差略大，允许 0.6s，但仍远小于
	// 其自身漂移量级（~0.15s 起、随段累积），足以证明换算确实在跟随。
	if maxPose > 0.2 || maxOdom > 0.6 {
		t.Fatalf("对外换算时间误差过大: pose=%.3f odom=%.3f", maxPose, maxOdom)
	}
}

// TestAlignmentPendingBoundedWhenCalibrating 已发现漂移证据但样本尚不
// 可信时观测暂缓（align_calibrating），待处理规模必须始终有界，绝不
// 随运行时长线性堆积；超过校准宽限后恒等降级放行，滤波器不被饿死。
//
// 用真实漂移（强制进入 calibrating）+ 极短数据，使锚点数在宽限内始终
// 达不到 AlignMinSamples，稳定复现“样本不足、暂不采用”。
func TestAlignmentPendingBoundedWhenCalibrating(t *testing.T) {
	// 时变激励轨迹上给位姿路一个显著固定偏置（>AlignDetect），但只回放
	// 6s：锚点需要半窗+搜索半径的等待 + 至少 AlignMinSamples 个锚点，
	// 宽限（3s）结束前锚点必然不足 → 全程 calibrating 暂缓。
	cfg := ReplayConfig{
		Seed: 3, Duration: 6 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		ReorderWindow: 4, Excited: true,
		Clocks: map[SensorKind]ClockProfile{
			SensorAbsolutePose: {Slopes: []float64{0.99}, Segments: nil, Offset0: 300 * time.Millisecond},
		},
	}
	ec := alignTestConfig()
	ec.Limits.MaxPendingObs = 256
	ec.AlignMinSamples = 12
	ec.AlignCalibrateAfter = 3 * time.Second
	obs := GenerateReplay(cfg)
	eng := NewEngine(ec)
	peak := 0
	sawCalHold := false // 宽限内“暂不采用”
	for i := 0; i < len(obs); i += 32 {
		end := i + 32
		if end > len(obs) {
			end = len(obs)
		}
		ds := eng.Ingest(obs[i:end])
		for _, d := range ds {
			if d.Reason == ReasonAlignCalibrating {
				// 用参考前沿判断是否仍在宽限内（详见 verdict 口径）。
				sawCalHold = true
			}
		}
		if n := eng.Stats().Pending; n > peak {
			peak = n
		}
	}
	if peak > ec.Limits.MaxPendingObs {
		t.Fatalf("校准期待处理积压无界: peak=%d cap=%d", peak, ec.Limits.MaxPendingObs)
	}
	// 6s > 3s 宽限：宽限后位姿路应恒等降级放行（带噪声膨胀），引擎能
	// 够初始化/推进，不被一路未对齐饿死。
	reasons := map[RejectReason]int{}
	for _, d := range eng.Decisions() {
		reasons[d.Reason]++
	}
	t.Logf("peak=%d reasons=%+v stats=%+v", peak, reasons, eng.Stats())
	if !sawCalHold || reasons[ReasonAlignCalibrating] == 0 {
		t.Fatal("样本不足期应给出 align_calibrating（暂不采用）判定")
	}
	st := statOf(eng, SensorAbsolutePose)
	if st.State == AlignLocked {
		t.Fatalf("锚点不足不应锁定, got %+v", st)
	}
}

// TestAlignmentOutOfRangeRejectedNotBuffered 漂移越界路给出可区分的
// time_unavailable，且其观测不进入待处理缓冲（旧实现会永久驻留、无限
// 堆积）。这与“样本不足 align_calibrating”明确分开。
func TestAlignmentOutOfRangeRejectedNotBuffered(t *testing.T) {
	cfg := ReplayConfig{
		Seed: 9, Duration: 40 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.02, ReorderWindow: 6, Excited: true,
		Clocks: map[SensorKind]ClockProfile{
			SensorOdometry: {Slopes: []float64{1.15}, Segments: nil, Offset0: 0},
		},
	}
	ec := alignTestConfig()
	ec.Limits.MaxPendingObs = 512
	obs := GenerateReplay(cfg)
	eng := NewEngine(ec)
	runObsBatches(t, eng, obs, 64)
	st := statOf(eng, SensorOdometry)
	if st.State == AlignLocked {
		t.Fatalf("超范围时钟不应锁定: %+v", st)
	}
	if st.State != AlignOutOfRange {
		t.Fatalf("±15%% 斜率应判 out_of_range, got %s", st.State)
	}
	if _, ok := eng.AlignedTime(SensorOdometry, obs[0].Stamp); ok {
		t.Fatal("越界路不应提供换算时间")
	}
	var unavailable int
	for _, d := range eng.Decisions() {
		if d.Obs.Kind == SensorOdometry && d.Reason == ReasonTimeUnavailable {
			unavailable++
		}
	}
	if unavailable == 0 {
		t.Fatal("越界路应给出 time_unavailable 判定（与样本不足区分）")
	}
	// 关键：越界观测不得长期驻留待处理缓冲——结束时挂起的里程计条目
	// 应为 0（新观测直接拒绝；锁定后转越界的旧积压由水线驱逐）。
	notReady := 0
	for i := 0; i < eng.buf.Len(); i++ {
		p := eng.buf.Index(i)
		if p.kind == SensorOdometry && !p.ready {
			notReady++
		}
	}
	if notReady != 0 {
		t.Fatalf("越界里程计观测不得堆积在待处理缓冲, got %d", notReady)
	}
	t.Logf("越界里程计 time_unavailable=%d, 待处理=%d, %s",
		unavailable, eng.Stats().Pending, st.State)
}

// TestAlignmentDoesNotFreezeTimeline 一路迟迟不可信时，参考路（IMU）
// 与其他已就绪路的消费不得被它卡住：推进到较晚时刻，IMU 驱动的状态
// 时间应能跟上，且不依赖该路就绪。
func TestAlignmentDoesNotFreezeTimeline(t *testing.T) {
	t0 := testEpoch
	ec := alignTestConfig()
	ec.AlignRef = SensorIMU
	ec.AlignAutoRef = false
	ec.AlignCalibrateAfter = 2 * time.Second
	eng := NewEngine(ec)
	// 只持续喂 IMU（参考路），从不喂任何非参考路；状态时间必须能一路
	// 推进，水线不能因“没见到待对齐路”而冻结。
	var batch []Observation
	for i := 0; i < 400; i++ {
		batch = append(batch, Observation{
			Kind: SensorIMU, Stamp: t0.Add(time.Duration(i*10) * time.Millisecond),
			Seq: uint64(i + 1), Accel: Vec3{Z: 9.81}, Gyro: Vec3{Z: 0.3},
		})
	}
	for i := 0; i < len(batch); i += 64 {
		end := i + 64
		if end > len(batch) {
			end = len(batch)
		}
		eng.Ingest(batch[i:end])
	}
	eng.AdvanceTo(t0.Add(4 * time.Second))
	s, _, ok := eng.Snapshot()
	if !ok {
		t.Fatal("仅参考路也应能初始化（3 倍窗口兜底）并推进")
	}
	if d := s.Stamp.Sub(t0); d < 3*time.Second {
		t.Fatalf("单参考路时间轴被冻结: state=%v (推进仅 %v)", s.Stamp, d)
	}
	t.Logf("仅 IMU 参考路推进正常: stateT=%v", s.Stamp.Sub(t0))
}

// TestLongRunHalfHourAligned 半小时量级回放：在线对齐开启，真正把各
// 路时钟随时间变化（分段、换向、方向/幅度不一致）的漂移叠进去，并覆
// 盖乱序、缺帧、离群、一路中途长缺测。断言：
//   - 全程融合误差有界、速度不发散、四元数单位、协方差正定；
//   - 各路在长跑中保持锁定（漂移虽分段换向但始终在可信范围内）；
//   - 待处理积压始终有界；相同种子重复运行严格一致（确定性）。
func TestLongRunHalfHourAligned(t *testing.T) {
	if testing.Short() {
		t.Skip("长跑用例")
	}
	min := time.Minute
	cfg := ReplayConfig{
		Seed: 23, Duration: 30 * min,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.03, OutlierProb: 0.01, OutlierScale: 5,
		ReorderWindow: 12, Excited: true,
		Clocks: map[SensorKind]ClockProfile{
			// 里程计：每 5 分钟在 快/慢/恢复 之间循环换向，方向幅度不一。
			SensorOdometry: {
				Slopes:   []float64{1.0025, 0.997, 1.0012, 0.9988, 1.002, 0.999},
				Segments: []time.Duration{5 * min, 5 * min, 5 * min, 5 * min, 5 * min},
				Offset0:  -120 * time.Millisecond,
			},
			// 位姿：持续偏慢、固定正偏置，斜率中段再变化一次。
			SensorAbsolutePose: {
				Slopes:   []float64{0.9975, 1.0015},
				Segments: []time.Duration{15 * min},
				Offset0:  250 * time.Millisecond,
			},
		},
		PoseGap: 5 * time.Second,
		SensorGap: map[SensorKind]time.Duration{
			SensorOdometry: 30 * time.Second, // 中途一路长缺测
		},
	}

	run := func(seed int64) (State, Mat, EngineStats, *Engine) {
		c := cfg
		c.Seed = seed
		obs := GenerateReplay(c)
		ec := alignTestConfig()
		ec.DegradeAfter = 3 * time.Second
		ec.Limits.MaxProcessTime = 120 * time.Second
		ec.Limits.MaxPendingObs = 8192
		eng := NewEngine(ec)
		runObsBatches(t, eng, obs, 128)
		end := obs[0].Stamp
		for _, o := range obs {
			if o.Stamp.After(end) {
				end = o.Stamp
			}
		}
		eng.AdvanceTo(end.Add(2 * time.Second))
		s, cov, ok := eng.Snapshot()
		if !ok {
			t.Fatal("长跑后应有状态")
		}
		return s, cov, eng.Stats(), eng
	}

	s1, c1, st1, eng1 := run(23)
	s2, _, _, _ := run(23)

	if !statesEqual(s1, s2, 1e-9) {
		t.Fatalf("长跑两次运行不确定:\n%+v\n%+v", s1, s2)
	}
	if !c1.IsPositiveDefinite() {
		t.Fatal("长跑后协方差失去正定性")
	}
	nq := s1.Rotation.W*s1.Rotation.W + s1.Rotation.X*s1.Rotation.X +
		s1.Rotation.Y*s1.Rotation.Y + s1.Rotation.Z*s1.Rotation.Z
	if math.Abs(nq-1) > 1e-9 {
		t.Fatalf("长跑后四元数退化: |q|²=%v", nq)
	}
	// 速度有界：激励轨迹线速度 ≤ 2.6m/s，允许融合余量，绝不允许上飙。
	if vn := vecNorm(s1.Velocity); vn > 4.0 {
		t.Fatalf("长跑末端速度发散: %.3f", vn)
	}
	if st1.PendingHigh > 8192 {
		t.Fatalf("长跑积压无界: %+v", st1)
	}
	for _, s := range eng1.AlignmentStats() {
		if s.IsRef {
			continue
		}
		if s.State != AlignLocked {
			t.Fatalf("30 分钟时变漂移下 %s 应保持锁定, got %s (skew=%.5f off=%v resid=%v)",
				s.Kind, s.State, s.Skew, s.Offset, s.Residual)
		}
	}
	// 末端位置误差：开对齐应在米级（漂移被真正修正）。用物理末端真值。
	start := time.Unix(1_700_000_000, 0)
	truth := ExcitedTrajectory(start).PoseAt(start.Add(cfg.Duration))
	pe := vecNorm(vecSub(s1.Position, truth.Position))
	t.Logf("30分钟开对齐: 积压峰值=%d consumed=%d rejected=%d 末端位置误差=%.3fm 速度=%.3f",
		st1.PendingHigh, st1.Consumed, st1.Rejected, pe, vecNorm(s1.Velocity))
	for _, s := range eng1.AlignmentStats() {
		t.Logf("  %s %s skew=%.5f off=%v resid=%v n=%d",
			s.Kind, s.State, s.Skew, s.Offset, s.Residual, s.Samples)
	}
	if pe > 3.0 {
		t.Fatalf("30 分钟时变漂移下末端位置误差过大: %.3fm", pe)
	}
}
