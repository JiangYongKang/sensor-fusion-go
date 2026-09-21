package fusion

import (
	"math"
	"testing"
)

func TestCholeskyAndSolve(t *testing.T) {
	m := MatDiag([]float64{4, 9, 16})
	m.Set(0, 1, 1)
	m.Set(1, 0, 1)
	L, ok := m.Cholesky()
	if !ok {
		t.Fatal("expected positive definite")
	}
	// 验证 L·Lᵀ = m
	rec := L.Mul(L.Transpose())
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(rec.At(i, j)-m.At(i, j)) > 1e-12 {
				t.Fatalf("reconstruction mismatch at (%d,%d)", i, j)
			}
		}
	}
	// 解 m·x = b
	b := []float64{1, 2, 3}
	x := SolveUpper(L, SolveLower(L, b))
	mv := m.MulVec(x)
	for i := range b {
		if math.Abs(mv[i]-b[i]) > 1e-9 {
			t.Fatalf("solve mismatch: got %v want %v", mv, b)
		}
	}
}

func TestPositiveDefiniteChecks(t *testing.T) {
	if !MatDiag([]float64{1, 2, 3}).IsPositiveDefinite() {
		t.Fatal("diag positive should be PD")
	}
	bad := MatDiag([]float64{1, -1, 3})
	if bad.IsPositiveDefinite() {
		t.Fatal("negative diag must not be PD")
	}
	// ClampDiagFloor 修复零对角。
	zero := MatDiag([]float64{1, 0, 3})
	zero.ClampDiagFloor(1e-9)
	if !zero.IsPositiveDefinite() {
		t.Fatal("clamped matrix should be PD")
	}
}

func TestSymmetrize(t *testing.T) {
	m := MatDiag([]float64{1, 1})
	m.Set(0, 1, 0.5)
	m.Set(1, 0, 0.7)
	m.Symmetrize()
	if m.At(0, 1) != 0.6 || m.At(1, 0) != 0.6 {
		t.Fatalf("symmetrize failed: %v %v", m.At(0, 1), m.At(1, 0))
	}
}
