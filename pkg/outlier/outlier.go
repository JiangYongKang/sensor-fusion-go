// Package outlier implements the measurement-credibility test applied
// before every observation update: a chi-square style Mahalanobis gate on
// the innovation with a configurable sigma/quantile threshold.
package outlier

import (
	"fmt"
	"math"

	"github.com/highcumontoa/sensor-fusion-go/pkg/linalg"
)

// Verdict is the gate decision.
type Verdict int

const (
	VerdictAccept Verdict = iota
	VerdictRejectGate
	VerdictRejectDegenerate // innovation covariance not positive definite
)

// Gate is a Mahalanobis-distance credibility gate.
type Gate struct {
	sigma  float64
	maxChi float64
}

// Config configures the gate.
type Config struct {
	Sigma  float64 // standard-normal quantile, e.g. 3.0 (chi2 via Wilson-Hilferty)
	MaxChi float64 // optional hard per-component innovation cutoff; <=0 disables
}

// New builds a gate with safe defaults.
func New(cfg Config) *Gate {
	if cfg.Sigma <= 0 {
		cfg.Sigma = 3
	}
	return &Gate{sigma: cfg.Sigma, maxChi: cfg.MaxChi}
}

// Threshold2 returns the squared-normal-sigma configuration.
func (g *Gate) Threshold2() float64 { return g.sigma * g.sigma }

// chiSquaredThreshold returns the chi-squared quantile with dof degrees of
// freedom at the two-sided normal quantile g.sigma, via the deterministic
// Wilson-Hilferty cube-root approximation (accurate to ~0.02 for dof>=1).
func (g *Gate) chiSquaredThreshold(dof int) float64 {
	n := float64(dof)
	z := g.sigma
	base := 1 - 2/(9*n) + z*math.Sqrt(2/(9*n))
	return n * base * base * base
}

// Test evaluates innovation ~ N(0, S). Returns d^2 = innov^T S^-1 innov.
func (g *Gate) Test(innov linalg.Vec, S linalg.Mat) (float64, Verdict, string) {
	Ss := S.Symmetrize()
	Sinv, ok := Ss.SolveSPD(innov)
	if !ok {
		return 0, VerdictRejectDegenerate, "innovation covariance is not positive definite"
	}
	d2 := innov.Dot(Sinv)
	if math.IsNaN(d2) || math.IsInf(d2, 0) {
		return 0, VerdictRejectDegenerate, "non-finite Mahalanobis distance"
	}
	thr := g.chiSquaredThreshold(innov.Len())
	if d2 > thr {
		return d2, VerdictRejectGate,
			fmt.Sprintf("d2=%.4f > chi2-thr=%.4f (sigma=%.2f,dof=%d)", d2, thr, g.sigma, innov.Len())
	}
	if g.maxChi > 0 {
		for i := 0; i < innov.Len(); i++ {
			if math.Abs(innov.At(i)) > g.maxChi {
				return d2, VerdictRejectGate,
					fmt.Sprintf("component %d |innov|=%.4f > max=%.4f", i, math.Abs(innov.At(i)), g.maxChi)
			}
		}
	}
	return d2, VerdictAccept,
		fmt.Sprintf("d2=%.4f <= chi2-thr=%.4f (sigma=%.2f,dof=%d)", d2, thr, g.sigma, innov.Len())
}
