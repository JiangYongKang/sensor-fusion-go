package fusion

import (
	"math"
	"testing"

	"github.com/highcumontoa/sensor-fusion-go/pkg/pose"
	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
)

func baseCfg() Config {
	return Config{
		InitialPosSigma: 1.0,
		InitialVelSigma: 1.0,
		InitialAttSigma: 0.2,
		Process:         Noise{Accel: 0.02, Gyro: 1e-4},
		Gravity:         [3]float64{0, 0, -9.81},
		MaxPositionJump: 10,
		GateSigma:       3,
		GateMaxChi:      0,
	}
}

func absObs(tNS int64, p [3]float64, q pose.Quat) sensors.Observation {
	return sensors.Observation{
		Stream:     sensors.StreamAbsPose,
		LocalNS:    tNS,
		NativeNS:   tNS,
		HasAbsPose: true,
		Pos:        p,
		Quat:       [4]float64{q.W, q.X, q.Y, q.Z},
		SigmaPos:   0.02,
		SigmaAng:   0.01,
	}
}

func TestAbsPoseConvergence(t *testing.T) {
	cfg := baseCfg()
	f := New(cfg, 0)
	truthP := [3]float64{1, -2, 0.5}
	truthQ := pose.QuatFromAxisAngle(pose.Vec(0, 0, 1), 0.25)
	for k := 1; k <= 40; k++ {
		tn := int64(k) * 10_000_000
		f.PredictTo(tn, nil)
		res := f.UpdateAbs(absObs(tn, truthP, truthQ))
		if !res.Applied {
			t.Fatalf("update %d rejected: %s", k, res.Detail)
		}
	}
	s := f.Snapshot()
	for i, want := range truthP {
		got := s.Pos.Slice()[i]
		if math.Abs(got-want) > 0.02 {
			t.Fatalf("pos[%d]=%.4f want %.4f", i, got, want)
		}
	}
	if ang := s.Quat.AngleTo(truthQ); ang > 0.01 {
		t.Fatalf("attitude err=%.5f rad", ang)
	}
	t.Logf("final pos=%v attErr=%.5f rad", s.Pos, s.Quat.AngleTo(truthQ))
}

func TestCovarianceStaysSPD(t *testing.T) {
	f := New(baseCfg(), 0)
	for k := 1; k <= 2000; k++ {
		tn := int64(k) * 1_000_000
		f.PredictTo(tn, nil)
		f.UpdateAbs(absObs(tn, [3]float64{0.1, 0.2, 0.3}, pose.QuatFromAxisAngle(pose.Vec(0, 0, 1), 0.01)))
		if _, ok := f.Snapshot().Cov.Cholesky(); !ok {
			t.Fatalf("covariance lost SPD at step %d", k)
		}
		if n := f.Snapshot().Quat.Norm(); math.Abs(n-1) > 1e-12 {
			t.Fatalf("quaternion norm %.15f at step %d", n, k)
		}
	}
}

func TestGateRejectDoesNotPollute(t *testing.T) {
	f := New(baseCfg(), 0)
	good := absObs(10_000_000, [3]float64{0, 0, 0}, pose.QuatIdentity())
	f.PredictTo(10_000_000, nil)
	r1 := f.UpdateAbs(good)
	if !r1.Applied {
		t.Fatal(r1.Detail)
	}
	before := f.Snapshot()
	bad := absObs(20_000_000, [3]float64{50, 0, 0}, pose.QuatIdentity())
	f.PredictTo(20_000_000, nil)
	r2 := f.UpdateAbs(bad)
	if r2.Applied {
		t.Fatal("50m outlier must be rejected")
	}
	t.Logf("outlier verdict=%d detail=%s", r2.Verdict, r2.Detail)
	after := f.Snapshot()
	if after.Pos != before.Pos {
		t.Fatal("state changed after gate rejection")
	}
}

func TestPredictRegressionRejected(t *testing.T) {
	f := New(baseCfg(), 1000)
	if !f.PredictTo(2000, nil) {
		t.Fatal("forward predict failed")
	}
	if f.PredictTo(1500, nil) {
		t.Fatal("backward predict must be rejected")
	}
	if f.TimeNS() != 2000 {
		t.Fatalf("time changed on regression: %d", f.TimeNS())
	}
}

func TestVelocityUpdate(t *testing.T) {
	cfg := baseCfg()
	f := New(cfg, 0)
	for k := 1; k <= 30; k++ {
		tn := int64(k) * 10_000_000
		f.PredictTo(tn, nil)
		o := sensors.Observation{Stream: sensors.StreamOdom, LocalNS: tn, NativeNS: tn,
			OdomV: [3]float64{1, 0, 0}, SigmaV: 0.02}
		r := f.UpdateVelocity(o)
		if !r.Applied {
			t.Fatalf("vel update %d: %s", k, r.Detail)
		}
	}
	v := f.Snapshot().Vel
	if math.Abs(v.X-1) > 0.05 || math.Abs(v.Y) > 0.05 || math.Abs(v.Z) > 0.05 {
		t.Fatalf("velocity = %v", v)
	}
}
