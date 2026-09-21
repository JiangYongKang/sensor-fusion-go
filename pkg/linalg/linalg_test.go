package linalg

import (
	"math"
	"testing"
)

func approx(t *testing.T, got, want, tol float64, name string) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("%s = %.12f, want %.12f", name, got, want)
	}
}

func TestSPDSolve(t *testing.T) {
	// A = L L^T with L = [[2 0 0],[1 2 0],[1 1 2]] -> SPD
	A := MatFrom(3, 3, []float64{
		4, 2, 2,
		2, 5, 3,
		2, 3, 6,
	})
	b := VecOf(1, 2, 3)
	x, ok := A.SolveSPD(b)
	if !ok {
		t.Fatal("SolveSPD failed on SPD matrix")
	}
	got := A.MulVec(x)
	for i := 0; i < 3; i++ {
		approx(t, got.At(i), b.At(i), 1e-12, "Ax=b component")
	}
	L, ok := A.Cholesky()
	if !ok {
		t.Fatal("Cholesky failed")
	}
	LLt := L.Mul(L.Transpose())
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			approx(t, LLt.At(i, j), A.At(i, j), 1e-12, "LL^T")
		}
	}
}

func TestCholeskyRejectsIndefinite(t *testing.T) {
	A := MatFrom(3, 3, []float64{
		1, 0, 0,
		0, -1, 0,
		0, 0, 1,
	})
	if _, ok := A.Cholesky(); ok {
		t.Fatal("expected Cholesky failure for indefinite matrix")
	}
}

func TestEigen3Diagonal(t *testing.T) {
	A := MatFrom(3, 3, []float64{
		7, 0, 0,
		0, 2, 0,
		0, 0, 5,
	})
	vals, _, ok := A.Eigen3()
	if !ok {
		t.Fatal("Eigen3 failed")
	}
	want := [3]float64{7, 5, 2}
	for i := range want {
		approx(t, vals[i], want[i], 1e-10, "eigenvalue")
	}
}

func TestEigen3Symmetric(t *testing.T) {
	// Random-ish symmetric matrix; verify A v = lambda v.
	A := MatFrom(3, 3, []float64{
		4, 1, 2,
		1, 3, 0.5,
		2, 0.5, 5,
	})
	vals, vecs, ok := A.Eigen3()
	if !ok {
		t.Fatal("Eigen3 failed")
	}
	t.Logf("eigenvalues = %v", vals)
	for k := 0; k < 3; k++ {
		v := VecOf(vecs[0][k], vecs[1][k], vecs[2][k])
		Av := A.MulVec(v)
		for i := 0; i < 3; i++ {
			approx(t, Av.At(i), vals[k]*v.At(i), 1e-9, "A v = lambda v")
		}
	}
	if vals[0] < vals[1] || vals[1] < vals[2] {
		t.Fatalf("eigenvalues not descending: %v", vals)
	}
}
