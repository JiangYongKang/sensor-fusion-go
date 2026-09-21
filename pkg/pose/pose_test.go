package pose

import (
	"math"
	"testing"
)

func approxF(t *testing.T, got, want, tol float64, name string) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("%s = %.12f want %.12f", name, got, want)
	}
}

func TestRotateZ90(t *testing.T) {
	q := QuatFromAxisAngle(Vec(0, 0, 1), math.Pi/2)
	v := q.Rotate(Vec(1, 0, 0))
	approxF(t, v.X, 0, 1e-12, "x")
	approxF(t, v.Y, 1, 1e-12, "y")
	approxF(t, v.Z, 0, 1e-12, "z")
	approxF(t, q.Norm(), 1, 1e-15, "|q|")
	t.Logf("Rz(90deg)*[1,0,0] = %v (q=%v)", v, q)
}

func TestComposition(t *testing.T) {
	a := QuatFromAxisAngle(Vec(0, 0, 1), 0.3)
	b := QuatFromAxisAngle(Vec(0, 0, 1), 0.4)
	c := a.Mul(b)
	want := QuatFromAxisAngle(Vec(0, 0, 1), 0.7)
	approxF(t, c.AngleTo(want), 0, 1e-12, "composed angle")
}

func TestExpLogRoundtrip(t *testing.T) {
	v := Vec(0.1, -0.2, 0.3)
	q := QuatExp(v)
	v2 := q.Log()
	approxF(t, v2.X, v.X, 1e-12, "rx")
	approxF(t, v2.Y, v.Y, 1e-12, "ry")
	approxF(t, v2.Z, v.Z, 1e-12, "rz")
}

func TestLongChainStaysUnit(t *testing.T) {
	// Integrate many tiny rotations; normalization each step must keep |q|=1.
	q := QuatIdentity()
	dq := QuatFromAxisAngle(Vec(0, 0, 1).Normalized(), 1e-4)
	for i := 0; i < 1_000_000; i++ {
		q = Normalize(q.Mul(dq))
	}
	approxF(t, q.Norm(), 1, 1e-12, "norm after 1e6 steps")
	want := QuatFromAxisAngle(Vec(0, 0, 1), 100.0)
	ang := q.AngleTo(want)
	approxF(t, ang, 0, 2e-6, "orientation after 1e6 steps")
	t.Logf("after 1e6 steps |q|=%.12f angular err=%.3e", q.Norm(), ang)
}

func TestDegenerateInputs(t *testing.T) {
	if Normalize(Quat{}) != QuatIdentity() {
		t.Fatal("zero quaternion must normalize to identity")
	}
	q := QuatFromAxisAngle(Vec3{}, 1.0)
	if q != QuatIdentity() {
		t.Fatal("zero axis must yield identity")
	}
	approxF(t, WrapAngle(3*math.Pi), -math.Pi, 1e-15, "wrap")
}
