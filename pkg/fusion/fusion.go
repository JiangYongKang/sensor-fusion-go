// Package fusion implements the time-propagated state estimator: an
// Error-State Kalman Filter over position, velocity and orientation
// (3-DOF rotation error) with process noise.
//
// State (nominal): p_W (3), v_W (3), q_WB (unit quaternion).
// Error state:    δx = [δp (3), δv (3), δθ_W (3)] (9 dims), with the
// convention q_true = exp(δθ_W) ⊗ q_est, so attitude innovations and
// corrections live in the world frame and no body/world Jacobian mixing is
// possible.
//
// Numerical-stability measures:
//   - orientation is a unit quaternion renormalised (with fixed hemisphere)
//     after every propagation and every correction;
//   - covariance is symmetrised and updated in Joseph form after every
//     measurement, then ridge-repaired to a strictly positive eigenvalue
//     floor;
//   - all linear solves go through Cholesky with explicit SPD checks, so a
//     degenerate innovation covariance can never silently corrupt state.
package fusion

import (
	"fmt"
	"math"

	"github.com/highcumontoa/sensor-fusion-go/pkg/linalg"
	"github.com/highcumontoa/sensor-fusion-go/pkg/outlier"
	"github.com/highcumontoa/sensor-fusion-go/pkg/pose"
	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
)

// StateDim is the error-state dimension (p, v, delta-theta).
const StateDim = 9

// Noise holds continuous-time process-noise intensities (PSD).
type Noise struct {
	Accel         float64 // (m/s^2)/sqrt(s)
	Gyro          float64 // rad/sqrt(s)
	AccelBiasWalk float64 // reserved for bias-state extension
}

// Config configures the filter.
type Config struct {
	InitialPosSigma float64
	InitialVelSigma float64
	InitialAttSigma float64
	Process         Noise
	Gravity         [3]float64 // m/s^2 in world frame
	MaxPositionJump float64    // hard sanity bound on correction, m
	GateSigma       float64    // Mahalanobis gate in normal sigmas
	GateMaxChi      float64    // hard per-component innovation cutoff (<=0 off)
	DefaultSigmaPos float64
	DefaultSigmaAng float64
	DefaultSigmaV   float64
	MinEigenvalue   float64 // covariance SPD floor
}

// Filter is the ESKF estimator. It is NOT goroutine-safe itself; the engine
// serialises all access.
type Filter struct {
	cfg  Config
	p    pose.Vec3
	v    pose.Vec3
	q    pose.Quat
	P    linalg.Mat
	tNS  int64
	step int64
}

// State is an immutable snapshot of the estimate.
type State struct {
	TNS  int64
	Pos  pose.Vec3
	Vel  pose.Vec3
	Quat pose.Quat
	Cov  linalg.Mat
	Step int64
}

// UpdateResult reports one measurement update.
type UpdateResult struct {
	Applied bool
	D2      float64
	Verdict outlier.Verdict
	Detail  string
	Innov   linalg.Vec
	S       linalg.Mat
}

// New builds a filter at t0NS with the configured initial uncertainty.
func New(cfg Config, t0NS int64) *Filter {
	cfg = withDefaults(cfg)
	f := &Filter{
		cfg: cfg,
		q:   pose.QuatIdentity(),
		P:   linalg.NewMat(StateDim, StateDim),
		tNS: t0NS,
	}
	for i := 0; i < 3; i++ {
		f.P.Set(i, i, cfg.InitialPosSigma*cfg.InitialPosSigma)
		f.P.Set(3+i, 3+i, cfg.InitialVelSigma*cfg.InitialVelSigma)
		f.P.Set(6+i, 6+i, cfg.InitialAttSigma*cfg.InitialAttSigma)
	}
	return f
}

func withDefaults(cfg Config) Config {
	d := func(v, def float64) float64 {
		if v <= 0 || math.IsNaN(v) {
			return def
		}
		return v
	}
	cfg.InitialPosSigma = d(cfg.InitialPosSigma, 1.0)
	cfg.InitialVelSigma = d(cfg.InitialVelSigma, 1.0)
	cfg.InitialAttSigma = d(cfg.InitialAttSigma, 0.1)
	cfg.Process.Accel = d(cfg.Process.Accel, 0.05)
	cfg.Process.Gyro = d(cfg.Process.Gyro, 1e-3)
	cfg.MaxPositionJump = d(cfg.MaxPositionJump, 50.0)
	cfg.GateSigma = d(cfg.GateSigma, 3.0)
	cfg.DefaultSigmaPos = d(cfg.DefaultSigmaPos, 0.05)
	cfg.DefaultSigmaAng = d(cfg.DefaultSigmaAng, 0.02)
	cfg.DefaultSigmaV = d(cfg.DefaultSigmaV, 0.1)
	cfg.MinEigenvalue = d(cfg.MinEigenvalue, 1e-12)
	if cfg.Gravity == [3]float64{} {
		cfg.Gravity = [3]float64{0, 0, -9.81}
	}
	return cfg
}

// Reset returns the filter to an identity estimate at t0NS.
func (f *Filter) Reset(cfg Config, t0NS int64) {
	*f = *New(cfg, t0NS)
}

// TimeNS is the last propagated-to time.
func (f *Filter) TimeNS() int64 { return f.tNS }

// PredictTo propagates state and covariance to tNS. lastIMU supplies the
// most recent accepted inertial sample; nil means "assume stationary"
// (world-frame acceleration and angular velocity zero). Returns false on a
// time regression, leaving state untouched.
func (f *Filter) PredictTo(tNS int64, lastIMU *sensors.Observation) bool {
	if tNS < f.tNS {
		return false
	}
	dt := float64(tNS-f.tNS) * 1e-9
	if dt > 0 {
		var aW, wB pose.Vec3
		if lastIMU != nil {
			aB := pose.VecFrom(lastIMU.Acc)
			wB = pose.VecFrom(lastIMU.Gyr)
			// Specific force: a_true_W = R_WB * f_b + g_W.
			aW = f.q.Rotate(aB).Add(pose.VecFrom(f.cfg.Gravity))
		}
		// Nominal propagation.
		f.p = f.p.Add(f.v.Scale(dt)).Add(aW.Scale(0.5 * dt * dt))
		f.v = f.v.Add(aW.Scale(dt))
		if wB.Norm() > 0 {
			f.q = pose.Normalize(f.q.Mul(pose.QuatExp(wB.Scale(dt))))
		} else {
			f.q = pose.Normalize(f.q)
		}
		f.propagateCovariance(dt)
	}
	f.tNS = tNS
	f.step++
	return true
}

func (f *Filter) propagateCovariance(dt float64) {
	Fm := linalg.Identity(StateDim)
	for i := 0; i < 3; i++ {
		Fm.Set(i, 3+i, dt) // δp += δv dt
	}
	sa2 := f.cfg.Process.Accel * f.cfg.Process.Accel
	sg2 := f.cfg.Process.Gyro * f.cfg.Process.Gyro
	// Discrete process covariance. Velocity is driven by white acceleration
	// noise; position inherits BOTH the integrated acceleration noise
	// (sa2*dt^3/3) and the velocity random-walk mapped over dt from the
	// pre-existing velocity uncertainty via F (handled by F P F^T), while
	// the injection noise below covers the white-noise channel. Keeping both
	// channels is essential for long gaps, where a small sa2*dt^3/3 alone
	// would leave position covariance vanishingly small.
	Q := linalg.NewMat(StateDim, StateDim)
	qpp := sa2 * dt * dt * dt / 3
	qpv := sa2 * dt * dt / 2
	qvv := sa2 * dt
	qww := sg2 * dt
	for i := 0; i < 3; i++ {
		Q.Set(i, i, qpp)
		Q.Set(i, 3+i, qpv)
		Q.Set(3+i, i, qpv)
		Q.Set(3+i, 3+i, qvv)
		Q.Set(6+i, 6+i, qww)
	}
	Pn := Fm.Mul(f.P).Mul(Fm.Transpose()).Add(Q)
	f.P = Pn.Symmetrize()
	f.enforceSPD(f.cfg.MinEigenvalue)
}

func sigma3(o *sensors.Observation, v, def float64) float64 {
	if !(v > 0) || math.IsNaN(v) {
		return def
	}
	return v
}

// UpdateAbs applies a 6-DOF absolute pose update.
func (f *Filter) UpdateAbs(o sensors.Observation) UpdateResult {
	if !o.HasAbsPose {
		return UpdateResult{Verdict: outlier.VerdictRejectDegenerate,
			Detail: "observation carries no absolute pose payload"}
	}
	qz := pose.Normalize(pose.Quat{W: o.Quat[0], X: o.Quat[1], Y: o.Quat[2], Z: o.Quat[3]})
	pz := pose.VecFrom(o.Pos)
	// Attitude innovation in world frame: Log(qz ⊗ q_est^{-1}).
	dq := qz.Mul(f.q.Conj()).Log()
	innov := linalg.NewVec(6)
	innovP := pz.Sub(f.p)
	for i := 0; i < 3; i++ {
		innov.Set(i, innovP.Slice()[i])
		innov.Set(3+i, dq.Slice()[i])
	}
	sp := sigma3(&o, o.SigmaPos, f.cfg.DefaultSigmaPos)
	sa := sigma3(&o, o.SigmaAng, f.cfg.DefaultSigmaAng)
	R := linalg.NewMat(6, 6)
	for i := 0; i < 3; i++ {
		R.Set(i, i, sp*sp)
		R.Set(3+i, 3+i, sa*sa)
	}
	H := linalg.Zeros(6, StateDim)
	for i := 0; i < 3; i++ {
		H.Set(i, i, 1)
		H.Set(3+i, 6+i, 1)
	}
	return f.applyUpdate(innov, H, R, true)
}

// UpdateVelocity applies a 3-DOF body-frame odometry velocity update. The
// measurement is rotated into the world frame so H remains a simple
// selector; its uncertainty is carried over isotropically.
func (f *Filter) UpdateVelocity(o sensors.Observation) UpdateResult {
	vb := pose.VecFrom(o.OdomV)
	vw := f.q.Rotate(vb)
	innov := linalg.NewVec(3)
	dv := vw.Sub(f.v)
	for i := 0; i < 3; i++ {
		innov.Set(i, dv.Slice()[i])
	}
	sv := sigma3(&o, o.SigmaV, f.cfg.DefaultSigmaV)
	R := linalg.NewMat(3, 3)
	for i := 0; i < 3; i++ {
		R.Set(i, i, sv*sv)
	}
	H := linalg.Zeros(3, StateDim)
	for i := 0; i < 3; i++ {
		H.Set(i, 3+i, 1)
	}
	return f.applyUpdate(innov, H, R, false)
}

func (f *Filter) applyUpdate(innov linalg.Vec, H, R linalg.Mat, isPose bool) UpdateResult {
	HP := H.Mul(f.P)
	S := HP.Mul(H.Transpose()).Add(R).Symmetrize()
	res := UpdateResult{Innov: innov, S: S}
	gate := outlier.New(outlier.Config{Sigma: f.cfg.GateSigma, MaxChi: f.cfg.GateMaxChi})
	d2, verdict, detail := gate.Test(innov, S)
	res.D2, res.Verdict, res.Detail = d2, verdict, detail
	if verdict != outlier.VerdictAccept {
		res.Applied = false
		return res
	}
	// K = P H^T S^{-1}, equivalently K^T = S^{-1} (H P): solve
	// S * Kt[:,i] = HP[:,i] for each column i of HP (6 x 9).
	HPm := H.Mul(f.P)
	n := innov.Len()
	Kt := linalg.NewMat(n, StateDim)
	for i := 0; i < StateDim; i++ {
		col := linalg.NewVec(n)
		for r := 0; r < n; r++ {
			col.Set(r, HPm.At(r, i))
		}
		sol, ok2 := S.SolveSPD(col)
		if !ok2 {
			res.Verdict = outlier.VerdictRejectDegenerate
			res.Detail = "S became singular during gain solve"
			return res
		}
		for r := 0; r < n; r++ {
			Kt.Set(r, i, sol.At(r))
		}
	}
	K := Kt.Transpose()
	dx := K.MulVec(innov)
	if isPose && f.cfg.MaxPositionJump > 0 {
		var dp2 float64
		for i := 0; i < 3; i++ {
			dp2 += dx.At(i) * dx.At(i)
		}
		if dp := math.Sqrt(dp2); dp > f.cfg.MaxPositionJump {
			res.Verdict = outlier.VerdictRejectGate
			res.Applied = false
			res.Detail = fmt.Sprintf("position correction %.4fm exceeds bound %.4fm (d2=%.3f)", dp, f.cfg.MaxPositionJump, d2)
			return res
		}
	}
	f.applyCorrection(dx)
	// Joseph covariance update: (I-KH) P (I-KH)^T + K R K^T.
	I := linalg.Identity(StateDim)
	KH := K.Mul(H)
	E := I.Sub(KH)
	Pn := E.Mul(f.P).Mul(E.Transpose()).Add(K.Mul(R).Mul(K.Transpose())).Symmetrize()
	f.P = Pn
	f.enforceSPD(f.cfg.MinEigenvalue)
	f.step++
	res.Applied = true
	res.Detail = fmt.Sprintf("applied: %s", detail)
	return res
}

func (f *Filter) applyCorrection(dx linalg.Vec) {
	f.p = f.p.Add(pose.Vec3{X: dx.At(0), Y: dx.At(1), Z: dx.At(2)})
	f.v = f.v.Add(pose.Vec3{X: dx.At(3), Y: dx.At(4), Z: dx.At(5)})
	dq := pose.QuatExp(pose.Vec3{X: dx.At(6), Y: dx.At(7), Z: dx.At(8)})
	f.q = pose.Normalize(dq.Mul(f.q))
}

// enforceSPD symmetrises and guarantees a strictly positive eigenvalue
// floor by adding an adaptive diagonal ridge whenever Cholesky fails.
func (f *Filter) enforceSPD(minEigen float64) {
	P := f.P.Symmetrize()
	ridge := minEigen
	for attempt := 0; attempt < 40; attempt++ {
		cand := P.Clone()
		if ridge > 0 {
			for i := 0; i < StateDim; i++ {
				cand.Set(i, i, cand.At(i, i)+ridge)
			}
		}
		if _, ok := cand.Cholesky(); ok {
			f.P = cand
			return
		}
		if ridge == 0 {
			ridge = minEigen
		} else {
			ridge *= 10
		}
	}
	// Last resort: reset to a conservative diagonal covariance.
	f.P = linalg.NewMat(StateDim, StateDim)
	for i := 0; i < StateDim; i++ {
		f.P.Set(i, i, 1.0)
	}
}

// Snapshot returns a defensive copy of the current estimate.
func (f *Filter) Snapshot() State {
	return State{
		TNS:  f.tNS,
		Pos:  f.p,
		Vel:  f.v,
		Quat: f.q,
		Cov:  f.P.Clone(),
		Step: f.step,
	}
}

// StepCount returns the number of predict/update steps performed.
func (f *Filter) StepCount() int64 { return f.step }
