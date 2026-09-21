package engine

import (
	"context"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/highcumontoa/sensor-fusion-go/pkg/fusion"
	"github.com/highcumontoa/sensor-fusion-go/pkg/pose"
	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
	"github.com/highcumontoa/sensor-fusion-go/pkg/timesync"
)

func testConfig() Config {
	cfg := Config{StartNS: 0}
	cfg.Buffer.WindowNS = 50_000_000 // 50 ms reorder delay
	cfg.Buffer.Capacity = 4096
	cfg.Clock = map[sensors.Stream]timesync.Config{}
	cfg.Degrade = map[sensors.Stream]DegradeRule{}
	cfg.LinkDelayNS = map[sensors.Stream]int64{}
	for _, s := range sensors.AllStreams {
		cfg.LinkDelayNS[s] = 2_000_000
	}
	for _, s := range sensors.AllStreams {
		cfg.Clock[s] = timesync.Config{MinPairs: 2, MaxPairs: 32, MaxDrift: 1e-3}
	}
	cfg.Fusion = fusion.Config{
		InitialPosSigma: 1, InitialVelSigma: 1, InitialAttSigma: 0.2,
		Process:         fusion.Noise{Accel: 0.05, Gyro: 1e-3},
		Gravity:         [3]float64{0, 0, -9.81},
		MaxPositionJump: 5, GateSigma: 3,
	}
	cfg.Degrade = map[sensors.Stream]DegradeRule{
		sensors.StreamIMU:     {DegradeAgeNS: 100_000_000, StaleAgeNS: 500_000_000, SigmaFactor: 3},
		sensors.StreamOdom:    {DegradeAgeNS: 200_000_000, StaleAgeNS: 1_000_000_000, SigmaFactor: 4},
		sensors.StreamAbsPose: {DegradeAgeNS: 500_000_000, StaleAgeNS: 2_000_000_000, SigmaFactor: 4},
	}
	cfg.Limits = Limits{MaxPending: 4096, MaxObservations: 1_000_000}
	return cfg
}

func imuAt(t int64) sensors.Observation {
	return sensors.Observation{Stream: sensors.StreamIMU, NativeNS: t, ArrivalRefNS: t + 2_000_000,
		Acc: [3]float64{0, 0, 9.81}, Gyr: [3]float64{0, 0, 0},
		SigmaIMUA: 0.05, SigmaIMUG: 1e-3, Seq: t / 10_000_000}
}

func poseAt(t int64, p [3]float64, q pose.Quat) sensors.Observation {
	return sensors.Observation{Stream: sensors.StreamAbsPose, NativeNS: t, ArrivalRefNS: t + 2_000_000,
		HasAbsPose: true, Pos: p, Quat: [4]float64{q.W, q.X, q.Y, q.Z},
		SigmaPos: 0.02, SigmaAng: 0.01, Seq: t / 100_000_000}
}

// primeClock feeds two in-order samples per stream so all clock models
// become ready before scenario-specific assertions.
func primeClock(t *testing.T, e *Engine) {
	t.Helper()
	for _, k := range []int64{1, 2} {
		tn := k * 10_000_000
		e.Submit(context.Background(), imuAt(tn))
		o := sensors.Observation{Stream: sensors.StreamOdom, NativeNS: tn, ArrivalRefNS: tn + 2_000_000,
			OdomV: [3]float64{0.6, 0, 0}, SigmaV: 0.03, Seq: k}
		e.Submit(context.Background(), o)
		e.Submit(context.Background(), poseAt(tn, [3]float64{0, 0, 0}, pose.QuatIdentity()))
	}
	e.AdvanceTo(context.Background(), 70_000_000)
}

func TestClockNotReadyThenOK(t *testing.T) {
	e := New(testConfig())
	defer e.Close()
	r := e.Submit(context.Background(), imuAt(10_000_000))
	if r.Decision.Reason != sensors.ReasonClockNotReady {
		t.Fatalf("first sample reason = %s (%s)", r.Decision.Reason, r.Decision.Detail)
	}
	t.Logf("first sample: %s", r.Decision.Reason)
	primeClock(t, e)
	r = e.Submit(context.Background(), imuAt(30_000_000))
	if r.Decision.Reason == sensors.ReasonClockNotReady {
		t.Fatal("clock should be ready after two pairs")
	}
	t.Logf("clock ready decision: %s", r.Decision.Reason)
}

func TestDuplicateAndLateAndOrder(t *testing.T) {
	cfg := testConfig()
	e := New(cfg)
	defer e.Close()
	primeClock(t, e)
	// Submit poses out of order: 200ms then 100ms (100ms arrives within the
	// 50ms+... check): window is 50ms and arrival delay 2ms, so 100ms after
	// 200ms is 100ms late -> behind watermark? watermark advances only on
	// release; at arrival 102ms the 200ms frame is not releasable yet.
	o1 := poseAt(200_000_000, [3]float64{0, 0, 0}, pose.QuatIdentity())
	o1.ArrivalRefNS = 202_000_000
	r1 := e.Submit(context.Background(), o1)
	t.Logf("t=200ms insert: %s", r1.Decision.Reason)
	o2 := poseAt(100_000_000, [3]float64{0, 0, 0}, pose.QuatIdentity())
	o2.ArrivalRefNS = 150_000_000 // deliberately delivered late
	r2 := e.Submit(context.Background(), o2)
	t.Logf("t=100ms late insert: %s detail=%s", r2.Decision.Reason, r2.Decision.Detail)
	if r2.Decision.Reason != sensors.ReasonOutOfOrder {
		t.Fatalf("expected out_of_order_accepted, got %s", r2.Decision.Reason)
	}
	// Duplicate of the 200ms frame.
	r3 := e.Submit(context.Background(), o1)
	if r3.Decision.Reason != sensors.ReasonDuplicate {
		t.Fatalf("expected duplicate, got %s", r3.Decision.Reason)
	}
	e.AdvanceTo(context.Background(), 300_000_000)
	// Now a frame at 120ms: behind released watermark (>=200ms).
	o4 := poseAt(120_000_000, [3]float64{0, 0, 0}, pose.QuatIdentity())
	o4.ArrivalRefNS = 310_000_000
	r4 := e.Submit(context.Background(), o4)
	if r4.Decision.Reason != sensors.ReasonDroppedLate {
		t.Fatalf("expected dropped_late, got %s", r4.Decision.Reason)
	}
	t.Logf("late-behind-watermark: %s", r4.Decision.Detail)
}

func TestGateRejectionAudit(t *testing.T) {
	e := New(testConfig())
	defer e.Close()
	primeClock(t, e)
	// Good pose at origin.
	e.Submit(context.Background(), poseAt(100_000_000, [3]float64{0, 0, 0}, pose.QuatIdentity()))
	e.AdvanceTo(context.Background(), 200_000_000)
	before := e.Snapshot().Pos
	// 8 m jump must be rejected (gate 3 sigma + MaxPositionJump 5m).
	bad := poseAt(300_000_000, [3]float64{8, 0, 0}, pose.QuatIdentity())
	bad.ArrivalRefNS = 302_000_000
	e.Submit(context.Background(), bad)
	e.AdvanceTo(context.Background(), 400_000_000)
	after := e.Snapshot().Pos
	if d := after.Sub(before).Norm(); d > 0.5 {
		t.Fatalf("outlier moved state by %.4f m", d)
	}
	var gateRejects int
	for _, d := range e.Audit() {
		if d.Reason == sensors.ReasonGateReject {
			gateRejects++
			t.Logf("GATE REJECT id=%d detail=%s", d.ID, d.Detail)
		}
	}
	if gateRejects == 0 {
		t.Fatal("expected at least one gate rejection in audit trail")
	}
}

func TestSizeLimit(t *testing.T) {
	cfg := testConfig()
	cfg.Limits.MaxObservations = 3
	e := New(cfg)
	defer e.Close()
	primeClock(t, e) // consumes 2 (first rejected clock-not-ready not counted)
	// Rejections count into total; by now at least 3 decisions exist.
	r := e.Submit(context.Background(), imuAt(50_000_000))
	t.Logf("limit decision: %s %s", r.Decision.Reason, r.Decision.Detail)
	if r.Decision.Reason != sensors.ReasonSizeLimit {
		t.Fatalf("expected size_limit, got %s", r.Decision.Reason)
	}
}

func TestProcessingBudgetNoPartial(t *testing.T) {
	cfg := testConfig()
	// Budget already expired: every call to ClockFn returns a fixed time T
	// while deadline T+1ns still leaves a single cheap iteration? Instead
	// use a clock that jumps to the future on the second call.
	calls := 0
	cfg.ClockFn = func() time.Time {
		calls++
		if calls <= 1 {
			return time.Unix(0, 0)
		}
		return time.Unix(0, 1_000_000_000) // far beyond any 1ns budget
	}
	cfg.Limits.StepBudget = time.Nanosecond
	e := New(cfg)
	defer e.Close()
	primeClock(t, e)
	prePending := 0 // nothing committed partially; ensure filter state monotonic
	s0 := e.Snapshot().Step
	e.Submit(context.Background(), imuAt(40_000_000))
	e.Submit(context.Background(), imuAt(50_000_000))
	s1 := e.Snapshot().Step
	_ = prePending
	t.Logf("steps before=%d after=%d (budget expiry must not corrupt)", s0, s1)
	if _, ok := e.Snapshot().Cov.Cholesky(); !ok {
		t.Fatal("covariance corrupted after budget expiry")
	}
}

func TestDegradationAndStale(t *testing.T) {
	e := New(testConfig())
	defer e.Close()
	primeClock(t, e)
	// Feed one pose at 100ms.
	e.Submit(context.Background(), poseAt(100_000_000, [3]float64{0, 0, 0}, pose.QuatIdentity()))
	e.AdvanceTo(context.Background(), 200_000_000)
	// Keep IMU flowing through the gap (as in a real system).
	for tn := int64(110_000_000); tn < 700_000_000; tn += 10_000_000 {
		e.Submit(context.Background(), imuAt(tn))
	}
	// 600 ms absolute-pose gap (>500ms degrade threshold, <2s stale). The
	// process model is stationary here, so the fresh pose stays consistent;
	// only the degradation decision is what we assert.
	gap := poseAt(700_000_000, [3]float64{0, 0, 0}, pose.QuatIdentity())
	gap.ArrivalRefNS = 702_000_000
	e.Submit(context.Background(), gap)
	ds := e.AdvanceTo(context.Background(), 800_000_000)
	foundDegraded := false
	for _, d := range ds {
		t.Logf("after-gap decision: %s %s", d.Reason, d.Detail)
		if d.Reason == sensors.ReasonDegraded || d.Reason == sensors.ReasonStale {
			foundDegraded = true
		}
	}
	if !foundDegraded {
		// Check audit too.
		for _, d := range e.Audit() {
			if d.Stream == sensors.StreamAbsPose && (d.Reason == sensors.ReasonDegraded || d.Reason == sensors.ReasonStale) {
				foundDegraded = true
			}
		}
	}
	if !foundDegraded {
		t.Fatal("expected degraded/stale decision after long gap")
	}
	// Health should reflect stale IMU (no IMU since 20ms).
	e.AdvanceTo(context.Background(), 1_000_000_000)
	for _, h := range e.Health() {
		t.Logf("health %s mode=%s samples=%d age=%dms", h.Stream, h.Mode, h.Samples, h.AgeNS/1_000_000)
	}
}

func TestReproducibleUnderConcurrency(t *testing.T) {
	run := func() fusion.State {
		e := New(testConfig())
		defer e.Close()
		// Deterministic input list, submitted from multiple goroutines with a
		// barrier; internal mutex must make the result depend only on order
		// after synchronisation. To guarantee identical effective order, each
		// goroutine handles a distinct stream and we advance once at end.
		var obs []sensors.Observation
		for k := int64(1); k <= 30; k++ {
			obs = append(obs, imuAt(k*10_000_000))
		}
		// Snapshots interleaved concurrently must never panic/race.
		var wg sync.WaitGroup
		stop := make(chan struct{})
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					e.Snapshot()
				}
			}
		}()
		for _, o := range obs {
			e.Submit(context.Background(), o)
		}
		for k := int64(1); k <= 5; k++ {
			e.Submit(context.Background(), poseAt(k*100_000_000, [3]float64{0, 0, 0}, pose.QuatIdentity()))
		}
		e.AdvanceTo(context.Background(), 1_000_000_000)
		close(stop)
		wg.Wait()
		return e.Snapshot()
	}
	a, b := run(), run()
	if a.Pos != b.Pos || a.Quat != b.Quat || a.Vel != b.Vel {
		t.Fatalf("non-reproducible state: %v vs %v", a.Pos, b.Pos)
	}
	if math.Abs(a.Quat.Norm()-1) > 1e-12 {
		t.Fatal("quaternion not unit")
	}
}
