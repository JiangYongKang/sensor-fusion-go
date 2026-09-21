// Package integration contains end-to-end tests that drive the engine with
// the deterministic replay: out-of-order delivery, dropped frames, injected
// outliers, per-stream clock drift and long runtime.
package integration

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/highcumontoa/sensor-fusion-go/pkg/engine"
	"github.com/highcumontoa/sensor-fusion-go/pkg/fusion"
	"github.com/highcumontoa/sensor-fusion-go/pkg/pose"
	"github.com/highcumontoa/sensor-fusion-go/pkg/replay"
	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
	"github.com/highcumontoa/sensor-fusion-go/pkg/timesync"
)

func engineConfig() engine.Config {
	var cfg engine.Config
	cfg.Buffer.WindowNS = 80_000_000
	cfg.Buffer.Capacity = 16_384
	cfg.Clock = map[sensors.Stream]timesync.Config{}
	cfg.LinkDelayNS = map[sensors.Stream]int64{}
	for _, s := range sensors.AllStreams {
		cfg.Clock[s] = timesync.Config{MinPairs: 2, MaxPairs: 64, MaxDrift: 1e-3}
		cfg.LinkDelayNS[s] = 2_000_000
	}
	cfg.Fusion = fusion.Config{
		InitialPosSigma: 1, InitialVelSigma: 1, InitialAttSigma: 0.2,
		Process:         fusion.Noise{Accel: 0.08, Gyro: 2e-3},
		Gravity:         [3]float64{0, 0, -9.81},
		MaxPositionJump: 8, GateSigma: 4,
		MinEigenvalue: 1e-12,
	}
	cfg.Degrade = map[sensors.Stream]engine.DegradeRule{
		sensors.StreamIMU:     {DegradeAgeNS: 100_000_000, StaleAgeNS: 1_000_000_000, SigmaFactor: 4},
		sensors.StreamOdom:    {DegradeAgeNS: 250_000_000, StaleAgeNS: 2_000_000_000, SigmaFactor: 4},
		sensors.StreamAbsPose: {DegradeAgeNS: 600_000_000, StaleAgeNS: 3_000_000_000, SigmaFactor: 4},
	}
	cfg.Limits = engine.Limits{MaxPending: 16_384, MaxObservations: 5_000_000}
	return cfg
}

type runStats struct {
	finalPos [3]float64
	finalQ   [4]float64
	posErr   float64
	attErr   float64
	maxPos   float64
	accepted int64
	rejected int64
	reasons  map[sensors.Reason]int
}

func runReplay(t *testing.T, seed int64, dur time.Duration) runStats {
	t.Helper()
	rc := replay.DefaultConfig()
	rc.Seed = seed
	rc.DurationNS = int64(dur / time.Nanosecond)
	gen := replay.New(rc)
	data := gen.Generate()
	e := engine.New(engineConfig())
	defer e.Close()

	st := runStats{reasons: map[sensors.Reason]int{}}
	ctx := context.Background()
	for _, o := range data.Order {
		rep := e.Submit(ctx, o)
		st.reasons[rep.Decision.Reason]++
		for _, d := range rep.Released {
			st.reasons[d.Reason]++
		}
		now := o.ArrivalRefNS
		ds := e.AdvanceTo(ctx, now)
		for _, d := range ds {
			st.reasons[d.Reason]++
		}
		s := e.Snapshot()
		if s.TNS > 1_000_000_000 { // evaluate after first second
			tr := gen.TruthAt(s.TNS)
			pe := pose.VecFrom(tr.Pos).Sub(s.Pos).Norm()
			if pe > st.maxPos {
				st.maxPos = pe
			}
		}
	}
	e.AdvanceTo(ctx, data.Order[len(data.Order)-1].ArrivalRefNS+1_000_000_000)
	snap := e.Snapshot()
	tr := gen.TruthAt(snap.TNS)
	st.posErr = pose.VecFrom(tr.Pos).Sub(snap.Pos).Norm()
	st.attErr = snap.Quat.AngleTo(pose.Quat{W: tr.Quat[0], X: tr.Quat[1], Y: tr.Quat[2], Z: tr.Quat[3]})
	st.finalPos = snap.Pos.Slice()
	st.finalQ = [4]float64{snap.Quat.W, snap.Quat.X, snap.Quat.Y, snap.Quat.Z}
	st.accepted = e.AcceptedCount()
	st.rejected = e.RejectedCount()
	if _, ok := snap.Cov.Cholesky(); !ok {
		t.Fatal("covariance not SPD at end of replay")
	}
	if math.Abs(snap.Quat.Norm()-1) > 1e-12 {
		t.Fatalf("quaternion norm %v", snap.Quat.Norm())
	}
	return st
}

func TestReplayAccuracyAndReasonCoverage(t *testing.T) {
	st := runReplay(t, 20260921, 20*time.Second)
	t.Logf("accepted=%d rejected=%d posErr=%.4fm attErr=%.5frad maxPosAfter1s=%.4fm",
		st.accepted, st.rejected, st.posErr, st.attErr, st.maxPos)
	for r, n := range st.reasons {
		t.Logf("reason %-32s %d", r, n)
	}
	if st.posErr > 0.15 {
		t.Fatalf("final position error too large: %.4f m", st.posErr)
	}
	if st.attErr > 0.08 {
		t.Fatalf("final attitude error too large: %.5f rad", st.attErr)
	}
	if st.maxPos > 0.25 {
		t.Fatalf("max position error too large: %.4f m", st.maxPos)
	}
	for _, want := range []sensors.Reason{
		sensors.ReasonOutOfOrder,
		sensors.ReasonDroppedLate,
		sensors.ReasonClockNotReady,
		sensors.ReasonGateReject,
	} {
		if st.reasons[want] == 0 {
			t.Errorf("expected at least one decision of reason %s", want)
		}
	}
	if st.reasons[sensors.ReasonGateReject] < 1 {
		t.Fatal("injected outliers were not gated")
	}
}

func TestReplayReproducible(t *testing.T) {
	a := runReplay(t, 777, 10*time.Second)
	b := runReplay(t, 777, 10*time.Second)
	if a.finalPos != b.finalPos || a.finalQ != b.finalQ {
		t.Fatalf("non-reproducible:\n%v %v\n%v %v", a.finalPos, a.finalQ, b.finalPos, b.finalQ)
	}
	if a.accepted != b.accepted || a.rejected != b.rejected {
		t.Fatalf("counts differ: %d/%d vs %d/%d", a.accepted, a.rejected, b.accepted, b.rejected)
	}
	t.Logf("reproducible final pos=%v q=%.8f... accepted=%d", a.finalPos, a.finalQ, a.accepted)
}

func TestLongRunStability(t *testing.T) {
	if testing.Short() {
		t.Skip("long run skipped in -short mode")
	}
	// 10 minutes: thousands of updates; numerical integrity must hold.
	st := runReplay(t, 424242, 600*time.Second)
	t.Logf("10-min run: posErr=%.4fm attErr=%.5frad maxPos=%.4fm accepted=%d",
		st.posErr, st.attErr, st.maxPos, st.accepted)
	// Error bounds widen linearly-ish; the key assertions are the SPD/unit
	// checks done inside runReplay, plus bounded divergence here.
	if math.IsNaN(st.posErr) || st.posErr > 2.0 {
		t.Fatalf("long run diverged: posErr=%v", st.posErr)
	}
}
