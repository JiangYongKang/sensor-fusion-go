// Package pose defines the geometric primitives (3D vectors, unit
// quaternions and rigid-body poses) and the coordinate conventions used
// across the engine.
//
// Coordinate convention:
//   - World frame W is a fixed, right-handed, right-handed ENU-like frame.
//   - Body frame B is rigidly attached to the moving platform.
//   - Pose T_WB expresses body motion relative to the world frame.
//   - Quaternions use the Hamilton convention, order (w, x, y, z),
//     and q rotates vectors from B to W: v_W = q ⊗ v_B ⊗ q*.
package pose

import (
	"fmt"
	"math"
)

// Vec3 is a 3D vector.
type Vec3 struct {
	X, Y, Z float64
}

// Quat is a Hamilton-convention unit quaternion {W, X, Y, Z}.
type Quat struct {
	W, X, Y, Z float64
}

// Pose is a rigid-body transform T_WB: position p_W and orientation q_WB.
type Pose struct {
	P Vec3
	Q Quat
}

// Twist is a body-frame velocity twist (linear + angular velocity).
type Twist struct {
	V Vec3 // linear velocity, m/s
	W Vec3 // angular velocity, rad/s
}

// Vec builds a Vec3 from 3 components (zero-padded).
func Vec(v ...float64) Vec3 {
	var a [3]float64
	copy(a[:], v)
	return VecFrom(a)
}

func (a Vec3) Add(b Vec3) Vec3      { return Vec3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func (a Vec3) Sub(b Vec3) Vec3      { return Vec3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }
func (a Vec3) Scale(s float64) Vec3 { return Vec3{a.X * s, a.Y * s, a.Z * s} }
func (a Vec3) Dot(b Vec3) float64   { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }

func (a Vec3) Cross(b Vec3) Vec3 {
	return Vec3{
		a.Y*b.Z - a.Z*b.Y,
		a.Z*b.X - a.X*b.Z,
		a.X*b.Y - a.Y*b.X,
	}
}

func (a Vec3) Norm() float64 { return math.Hypot(math.Hypot(a.X, a.Y), a.Z) }

// Normalized returns a unit vector. A zero vector maps deterministically to
// the X axis rather than NaN.
func (a Vec3) Normalized() Vec3 {
	n := a.Norm()
	if n == 0 || math.IsNaN(n) {
		return Vec3{X: 1}
	}
	return Vec3{a.X / n, a.Y / n, a.Z / n}
}

func (a Vec3) Slice() [3]float64 { return [3]float64{a.X, a.Y, a.Z} }

func VecFrom(a [3]float64) Vec3 { return Vec3{a[0], a[1], a[2]} }

// QuatIdentity is the body-to-world identity rotation.
func QuatIdentity() Quat { return Quat{W: 1} }

// QuatFromAxisAngle builds q = cos(a/2) + sin(a/2)*(ux,uy,uz).
// A non-finite or zero axis yields identity deterministically.
func QuatFromAxisAngle(axis Vec3, angle float64) Quat {
	if math.IsNaN(angle) || math.IsInf(angle, 0) {
		return QuatIdentity()
	}
	n := axis.Norm()
	if n == 0 || math.IsNaN(n) {
		return QuatIdentity()
	}
	h := angle * 0.5
	s := math.Sin(h) / n
	return Normalize(Quat{W: math.Cos(h), X: axis.X * s, Y: axis.Y * s, Z: axis.Z * s})
}

// Normalize returns a unit quaternion, choosing the hemisphere with a
// non-negative scalar part to keep interpolation/integration continuous.
// Degenerate input maps deterministically to identity.
func Normalize(q Quat) Quat {
	n := math.Sqrt(q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z)
	if n == 0 || math.IsNaN(n) || math.IsInf(n, 0) {
		return QuatIdentity()
	}
	q = Quat{q.W / n, q.X / n, q.Y / n, q.Z / n}
	if q.W < 0 {
		q.W, q.X, q.Y, q.Z = -q.W, -q.X, -q.Y, -q.Z
	}
	return q
}

// Rotate rotates a body-frame vector into the world frame: q ⊗ v ⊗ q*.
func (q Quat) Rotate(v Vec3) Vec3 {
	// v' = v + 2 * qv × (qv × v + qw * v)
	qv := Vec3{q.X, q.Y, q.Z}
	t := qv.Cross(v).Scale(2)
	return v.Add(t.Scale(q.W)).Add(qv.Cross(t))
}

// Mul returns the Hamilton product q ⊗ o (apply o first, then q).
func (q Quat) Mul(o Quat) Quat {
	return Normalize(Quat{
		W: q.W*o.W - q.X*o.X - q.Y*o.Y - q.Z*o.Z,
		X: q.W*o.X + q.X*o.W + q.Y*o.Z - q.Z*o.Y,
		Y: q.W*o.Y - q.X*o.Z + q.Y*o.W + q.Z*o.X,
		Z: q.W*o.Z + q.X*o.Y - q.Y*o.X + q.Z*o.W,
	})
}

// Conj returns the conjugate quaternion q*.
func (q Quat) Conj() Quat { return Quat{q.W, -q.X, -q.Y, -q.Z} }

func (q Quat) Norm() float64 {
	return math.Sqrt(q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z)
}

// AngleTo returns the geodesic rotation angle in [0, pi] between two
// unit quaternions.
func (q Quat) AngleTo(o Quat) float64 {
	a := Normalize(q)
	b := Normalize(o)
	d := a.W*b.W + a.X*b.X + a.Y*b.Y + a.Z*b.Z
	if d > 1 {
		d = 1
	}
	if d < -1 {
		d = -1
	}
	return 2 * math.Acos(math.Abs(d))
}

// RotationMatrix returns R_WB (column vectors are rotated body axes).
func (q Quat) RotationMatrix() *[3][3]float64 {
	q = Normalize(q)
	w, x, y, z := q.W, q.X, q.Y, q.Z
	return &[3][3]float64{
		{1 - 2*(y*y+z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)},
		{2 * (x*y + w*z), 1 - 2*(x*x+z*z), 2 * (y*z - w*x)},
		{2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2*(x*x+y*y)},
	}
}

// Log returns the rotation vector theta such that q = exp(theta/2),
// |theta| in [0, pi].
func (q Quat) Log() Vec3 {
	q = Normalize(q)
	vn := math.Sqrt(q.X*q.X + q.Y*q.Y + q.Z*q.Z)
	if vn < 1e-12 {
		return Vec3{}
	}
	// Guard acos against rounding drift.
	w := q.W
	if w > 1 {
		w = 1
	} else if w < -1 {
		w = -1
	}
	angle := 2 * math.Acos(w)
	s := angle / vn
	return Vec3{q.X * s, q.Y * s, q.Z * s}
}

// QuatExp is the inverse of Log: unit quaternion from a rotation vector.
func QuatExp(v Vec3) Quat {
	theta := v.Norm()
	if theta < 1e-12 {
		return QuatIdentity()
	}
	h := theta * 0.5
	s := math.Sin(h) / theta
	return Normalize(Quat{W: math.Cos(h), X: v.X * s, Y: v.Y * s, Z: v.Z * s})
}

func (p Pose) String() string {
	return fmt.Sprintf("T_WB{p=(%.6f %.6f %.6f) q=(%.6f %.6f %.6f %.6f)}",
		p.P.X, p.P.Y, p.P.Z, p.Q.W, p.Q.X, p.Q.Y, p.Q.Z)
}

// WrapAngle maps an angle to [-pi, pi) deterministically.
func WrapAngle(a float64) float64 {
	a = math.Mod(a+math.Pi, 2*math.Pi)
	if a < 0 {
		a += 2 * math.Pi
	}
	return a - math.Pi
}
