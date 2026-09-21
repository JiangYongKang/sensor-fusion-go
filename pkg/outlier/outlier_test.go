package outlier

import (
	"testing"

	"github.com/highcumontoa/sensor-fusion-go/pkg/linalg"
)

func TestGateAcceptReject(t *testing.T) {
	g := New(Config{Sigma: 3, MaxChi: 100})
	S := linalg.MatFrom(2, 2, []float64{1, 0, 0, 1})
	// 2-sigma-ish residual -> accept.
	in := linalg.VecOf(2, 1)
	d2, v, detail := g.Test(in, S)
	t.Logf("small innov: d2=%.3f verdict=%d (%s)", d2, v, detail)
	if v != VerdictAccept {
		t.Fatal("expected accept")
	}
	// 5-sigma residual -> reject.
	in = linalg.VecOf(5, 0)
	d2, v, detail = g.Test(in, S)
	t.Logf("large innov: d2=%.3f verdict=%d (%s)", d2, v, detail)
	if v != VerdictRejectGate {
		t.Fatal("expected gate reject")
	}
}

func TestGateDegenerateCovariance(t *testing.T) {
	g := New(Config{Sigma: 3})
	S := linalg.MatFrom(2, 2, []float64{1, 1, 1, 1})
	_, v, _ := g.Test(linalg.VecOf(0.1, 0.1), S)
	if v != VerdictRejectDegenerate {
		t.Fatalf("expected degenerate verdict, got %d", v)
	}
}

func TestHardCutoff(t *testing.T) {
	g := New(Config{Sigma: 5, MaxChi: 1.0})
	S := linalg.MatFrom(1, 1, []float64{1})
	_, v, detail := g.Test(linalg.VecOf(2), S)
	if v != VerdictRejectGate {
		t.Fatalf("expected cutoff reject, got %d", v)
	}
	t.Log(detail)
}
