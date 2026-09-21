package fusion

import (
	"math"
	"testing"
)

func TestQuatNormStability(t *testing.T) {
	// 全零四元数归一化必须回退到单位四元数，不得产生 NaN。
	q := QuatNorm(Quat{})
	if math.IsNaN(q.W) || q.W != 1 {
		t.Fatalf("zero norm: got %+v, want identity", q)
	}
	// 极小但非零的四元数应被正确归一化（缩放路径不丢精度）。
	q = QuatNorm(Quat{W: 1e-300, X: 1e-301})
	if n := q.W*q.W + q.X*q.X; math.Abs(n-1) > 1e-9 {
		t.Fatalf("tiny norm: |q|^2=%v", n)
	}
	// 大数归一化不溢出。
	q = QuatNorm(Quat{W: 1e300, X: 1e300})
	if n := q.W*q.W + q.X*q.X; math.Abs(n-1) > 1e-9 {
		t.Fatalf("large norm: |q|^2=%v", n)
	}
}

func TestRotVecRoundTrip(t *testing.T) {
	cases := []Vec3{
		{0.1, -0.2, 0.3},
		{1e-10, 0, 0},          // 小角度
		{0, 0, math.Pi - 1e-6}, // 接近 π
		{0, 0, 0},              // 零旋转
	}
	for _, w := range cases {
		q := QuatFromRotVec(w)
		got := QuatToRotVec(q)
		if d := vecNorm(vecSub(got, w)); d > 1e-9 {
			t.Fatalf("roundtrip %v: got %v (diff %g)", w, got, d)
		}
	}
}

func TestQuatRotateConsistency(t *testing.T) {
	// 旋转矩阵与四元数旋转结果一致。
	q := QuatFromRotVec(Vec3{0.3, -0.5, 0.8})
	v := Vec3{1, 2, 3}
	got := QuatRotate(q, v)
	m := quatToRotMat(q)
	want := Vec3{
		X: m.At(0, 0)*v.X + m.At(0, 1)*v.Y + m.At(0, 2)*v.Z,
		Y: m.At(1, 0)*v.X + m.At(1, 1)*v.Y + m.At(1, 2)*v.Z,
		Z: m.At(2, 0)*v.X + m.At(2, 1)*v.Y + m.At(2, 2)*v.Z,
	}
	if d := vecNorm(vecSub(got, want)); d > 1e-12 {
		t.Fatalf("rotate mismatch: got %v want %v", got, want)
	}
	// 旋转保持模长。
	if math.Abs(vecNorm(got)-vecNorm(v)) > 1e-12 {
		t.Fatalf("rotation changed norm: %v -> %v", vecNorm(v), vecNorm(got))
	}
}

func TestSlerpEndpointsAndMidpoint(t *testing.T) {
	a := QuatIdentity()
	b := QuatFromRotVec(Vec3{Z: math.Pi / 2})
	if q := QuatSlerp(a, b, 0); vecNorm(vecSub(QuatToRotVec(q), Vec3{})) > 1e-9 {
		t.Fatalf("slerp(0) != a")
	}
	if q := QuatSlerp(a, b, 1); vecNorm(vecSub(QuatToRotVec(q), Vec3{Z: math.Pi / 2})) > 1e-9 {
		t.Fatalf("slerp(1) != b")
	}
	mid := QuatToRotVec(QuatSlerp(a, b, 0.5))
	if math.Abs(mid.Z-math.Pi/4) > 1e-9 {
		t.Fatalf("slerp midpoint yaw=%v want pi/4", mid.Z)
	}
	// 双覆盖：-b 与 b 表示同一旋转，插值应取短路径。
	nb := Quat{-b.W, -b.X, -b.Y, -b.Z}
	mid2 := QuatToRotVec(QuatSlerp(a, nb, 0.5))
	if math.Abs(mid2.Z-math.Pi/4) > 1e-9 {
		t.Fatalf("slerp double-cover midpoint yaw=%v want pi/4", mid2.Z)
	}
}

func TestLongIntegrationNoDegeneration(t *testing.T) {
	// 长时间积分：10 万步小旋转，姿态不得退化（保持单位范数与有限值）。
	q := QuatIdentity()
	dq := QuatFromRotVec(Vec3{Z: 1e-4})
	for i := 0; i < 100000; i++ {
		q = QuatNorm(QuatMul(q, dq))
	}
	n := q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z
	if math.Abs(n-1) > 1e-9 || math.IsNaN(n) {
		t.Fatalf("quaternion degenerated after 1e5 steps: |q|^2=%v", n)
	}
	// 总旋转角应为 10 rad mod 2π。
	total := QuatToRotVec(q)
	want := math.Mod(100000*1e-4, 2*math.Pi)
	if want > math.Pi {
		want -= 2 * math.Pi
	}
	if math.Abs(total.Z-want) > 1e-3 {
		t.Fatalf("accumulated yaw=%v want≈%v", total.Z, want)
	}
}
