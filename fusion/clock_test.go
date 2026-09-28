package fusion

import (
	"math"
	"testing"
	"time"
)

func boolPtr(b bool) *bool { return &b }

func alignTestConfig() Config {
	cfg := testConfig()
	cfg.AlignMinSamples = 20
	cfg.AlignConfirmSamples = 6
	cfg.AlignActivateAt = 30 * time.Millisecond
	cfg.AlignMaxOffset = 2 * time.Second
	cfg.AlignMaxSkew = 5000e-6
	cfg.AlignForgetting = 0.98
	cfg.AlignInitStd = 2 * time.Second
	cfg.AlignMeasStd = 10 * time.Millisecond
	return cfg
}

// feedClock 以固定物理间隔给某路喂“无噪声”时钟样本：
// 传感器读数 src，统一时间 uni=src（对齐模型应学到真实 b、s）。
func feedClock(t *testing.T, a *ClockAligner, k SensorKind, anchor time.Time,
	offsetSec, skew float64, n int, dtSec float64) {
	t.Helper()
	a.ensure(k, anchor, true) // 模拟初始化后才出现 → calibrating
	for i := 1; i <= n; i++ {
		src := float64(i) * dtSec
		// 设备读数 stamp = anchor+src；真值物理时刻 = anchor+(src-offset)/（1+skew)
		uni := (src - offsetSec) / (1 + skew)
		ph := a.Observe(k, clockSample{src: src, uni: uni, measVar: 1e-4})
		if i == n {
			t.Logf("喂入 %d 样本后阶段=%s offset*=%.4fs skew*=%.4f",
				n, ph, a.Status(k).Offset.Seconds(), a.Status(k).Skew)
		}
	}
}

// TestClockLearnsConstantOffset 固定偏移必须在样本充足后进入 active，
// 且换算时间回到统一时间轴。
func TestClockLearnsConstantOffset(t *testing.T) {
	cfg := alignTestConfig()
	a := NewClockAligner(cfg)
	anchor := testEpoch
	// IMU 参考路恒等。
	if st := a.Status(SensorIMU); !st.Trusted || st.Mapped {
		t.Fatalf("IMU 应为参考恒等: %+v", st)
	}
	const off = 0.3
	feedClock(t, a, SensorAbsolutePose, anchor, off, 0, 60, 0.05)
	st := a.Status(SensorAbsolutePose)
	if st.Phase != "active" {
		t.Fatalf("固定偏移应收敛为 active, got %s (samples=%d)", st.Phase, st.Samples)
	}
	// 读数比物理时刻快 off：映射修正量应接近 -off。
	if math.Abs(st.Offset.Seconds()+off) > 0.03 {
		t.Fatalf("偏移修正估计偏差过大: got %.4f want %.4f", st.Offset.Seconds(), -off)
	}
	// 换算：设备快 0.3s（读数=t+0.3），映射后应回到 t。
	stamp := anchor.Add(time.Second)
	mapped, trusted := a.Align(SensorAbsolutePose, stamp)
	if !trusted {
		t.Fatal("active 路映射必须可信")
	}
	got := mapped.Sub(anchor).Seconds()
	if math.Abs(got-(1-off)) > 0.03 {
		t.Fatalf("换算时间未对齐: got %.4f want %.4f", got, 1-off)
	}
}

// TestClockCalibratingDistinguished 样本不足时必须明确区分为
// align_calibrating（暂不采用），且与漂移越界 align_out_of_range 不同。
func TestClockCalibratingDistinguished(t *testing.T) {
	cfg := alignTestConfig()
	a := NewClockAligner(cfg)
	a.ensure(SensorOdometry, testEpoch, true)
	if a.Phase(SensorOdometry) != phaseCalibrating {
		t.Fatal("新出现的非参考路应先 calibrating")
	}
	if _, trusted := a.Align(SensorOdometry, testEpoch); trusted {
		t.Fatal("calibrating 路换算必须不可信")
	}
	feedClock(t, a, SensorOdometry, testEpoch, 0.3, 0, 5, 0.05)
	if a.Phase(SensorOdometry) != phaseCalibrating {
		t.Fatalf("样本不足必须保持 calibrating, got %s", a.Phase(SensorOdometry))
	}
}

// TestClockTracksTimeVaryingSkew 已激活模型的漂移速率在中途改变方向，
// 二维 RLS 必须跟踪得上。
func TestClockTracksTimeVaryingSkew(t *testing.T) {
	cfg := alignTestConfig()
	a := NewClockAligner(cfg)
	anchor := testEpoch
	m := a.ensure(SensorAbsolutePose, anchor, false)
	m.phase = phaseActive // 直接进入 active，测斜率跟踪
	m.n, m.confirm = 20, 20
	feed := func(from, to int, skew float64) {
		for i := from; i < to; i++ {
			sx := float64(i) * 0.5
			uni := sx / (1 + skew)
			a.Observe(SensorAbsolutePose, clockSample{src: sx, uni: uni, measVar: 1e-4})
		}
	}
	// 第一段：设备快 2000ppm（估计斜率应 <1）。
	feed(1, 120, 2000e-6)
	skew1 := a.Status(SensorAbsolutePose).Skew
	t.Logf("第一段 斜率偏差=%.1fppm (设备快 ⇒ 估计斜率<1)", skew1*1e6)
	if math.Abs(skew1+2000e-6) > 800e-6 {
		t.Fatalf("第一段频偏估计偏差: %.1fppm", skew1*1e6)
	}
	// 第二段：设备慢 1500ppm（方向反转，斜率应变正）。
	feed(120, 280, -1500e-6)
	skew2 := a.Status(SensorAbsolutePose).Skew
	t.Logf("第二段 斜率偏差=%.1fppm (设备慢 ⇒ 估计斜率>1)", skew2*1e6)
	if skew2 < 200e-6 {
		t.Fatalf("反转后斜率偏差必须变为正并跟踪, got %.1fppm", skew2*1e6)
	}
	if a.Phase(SensorAbsolutePose) != phaseActive {
		t.Fatalf("时变跟踪中应保持 active, got %s", a.Phase(SensorAbsolutePose))
	}
}

// TestClockOutOfRangeSticky 偏移/频偏超可信范围必须转入粘滞的
// out_of_range，换算不可信，且之后即使来正常样本也不自动恢复。
func TestClockOutOfRangeSticky(t *testing.T) {
	cfg := alignTestConfig()
	cfg.AlignMaxOffset = 500 * time.Millisecond
	a := NewClockAligner(cfg)
	anchor := testEpoch
	feedClock(t, a, SensorOdometry, anchor, 1.0, 0, 60, 0.05)
	if a.Phase(SensorOdometry) != phaseOutOfRange {
		t.Fatalf("偏移越界应 out_of_range, got %s", a.Phase(SensorOdometry))
	}
	if _, trusted := a.Align(SensorOdometry, anchor); trusted {
		t.Fatal("out_of_range 路换算必须不可信")
	}
	detail := a.OutOfRangeDetail(SensorOdometry)
	t.Logf("越界原因: %s", detail)
	// 再来零偏移样本也不得自动恢复。
	for i := 0; i < 40; i++ {
		a.Observe(SensorOdometry, clockSample{src: 5 + float64(i)*0.05, uni: 5 + float64(i)*0.05, measVar: 1e-4})
	}
	if a.Phase(SensorOdometry) != phaseOutOfRange {
		t.Fatal("越界状态必须粘滞，需重建引擎才能恢复")
	}
}

// TestClockZeroDriftIdentity 无漂移输入必须永远走恒等映射，
// 不进入 active（保证与旧版本逐位一致的处理路径）。
func TestClockZeroDriftIdentity(t *testing.T) {
	cfg := alignTestConfig()
	a := NewClockAligner(cfg)
	anchor := testEpoch
	a.ensure(SensorAbsolutePose, anchor, false) // 初始化前出现：provisional
	for i := 1; i <= 200; i++ {
		src := float64(i) * 0.05
		a.Observe(SensorAbsolutePose, clockSample{src: src, uni: src, measVar: 1e-4})
	}
	st := a.Status(SensorAbsolutePose)
	if st.Mapped || st.Phase == "active" {
		t.Fatalf("无漂移不得启用估计映射: phase=%s", st.Phase)
	}
	stamp := anchor.Add(3 * time.Second)
	mapped, trusted := a.Align(SensorAbsolutePose, stamp)
	if !trusted || !mapped.Equal(stamp) {
		t.Fatalf("无漂移必须逐位恒等: mapped=%v trusted=%v", mapped, trusted)
	}
}

// TestClockDisabled 显式关闭对齐时所有路均为参考恒等。
func TestClockDisabled(t *testing.T) {
	cfg := alignTestConfig()
	cfg.AlignEnabled = boolPtr(false)
	a := NewClockAligner(cfg)
	a.ensure(SensorAbsolutePose, testEpoch, true)
	if a.Phase(SensorAbsolutePose) != phaseReference {
		t.Fatalf("关闭对齐后应恒等参考, got %s", a.Phase(SensorAbsolutePose))
	}
}
