package fusion

import "math"

// QuatIdentity 单位四元数。
func QuatIdentity() Quat { return Quat{W: 1} }

// QuatNorm 归一化；近零范数时返回单位四元数保证数值稳定。
// 先按最大分量缩放，避免大数平方溢出或小数平方下溢。
func QuatNorm(q Quat) Quat {
	m := math.Max(math.Max(math.Abs(q.W), math.Abs(q.X)), math.Max(math.Abs(q.Y), math.Abs(q.Z)))
	if m < 1e-300 || math.IsNaN(m) || math.IsInf(m, 0) {
		return QuatIdentity()
	}
	s := Quat{W: q.W / m, X: q.X / m, Y: q.Y / m, Z: q.Z / m}
	n := math.Sqrt(s.W*s.W + s.X*s.X + s.Y*s.Y + s.Z*s.Z)
	inv := 1.0 / n
	return Quat{W: s.W * inv, X: s.X * inv, Y: s.Y * inv, Z: s.Z * inv}
}

// QuatMul 四元数乘法 q = a ⊗ b（Hamilton 积，先 a 后 b）。
func QuatMul(a, b Quat) Quat {
	return Quat{
		W: a.W*b.W - a.X*b.X - a.Y*b.Y - a.Z*b.Z,
		X: a.W*b.X + a.X*b.W + a.Y*b.Z - a.Z*b.Y,
		Y: a.W*b.Y - a.X*b.Z + a.Y*b.W + a.Z*b.X,
		Z: a.W*b.Z + a.X*b.Y - a.Y*b.X + a.Z*b.W,
	}
}

// QuatConj 共轭（单位四元数即逆）。
func QuatConj(q Quat) Quat { return Quat{W: q.W, X: -q.X, Y: -q.Y, Z: -q.Z} }

// QuatRotate 用四元数旋转向量 v' = q ⊗ v ⊗ q*。
func QuatRotate(q Quat, v Vec3) Vec3 {
	// 展开为向量形式：v' = v + 2w(u×v) + 2u×(u×v)
	u := Vec3{q.X, q.Y, q.Z}
	uv := vecCross(u, v)
	uuv := vecCross(u, uv)
	return vecAdd(v, vecScale(vecAdd(vecScale(uv, q.W), uuv), 2))
}

// QuatFromRotVec 由旋转向量（轴角，模长为弧度）构造四元数，小角度安全。
func QuatFromRotVec(w Vec3) Quat {
	theta := vecNorm(w)
	if theta < 1e-8 {
		// 二阶泰勒近似，避免除零
		return QuatNorm(Quat{W: 1, X: w.X / 2, Y: w.Y / 2, Z: w.Z / 2})
	}
	half := theta / 2
	s := math.Sin(half) / theta
	return QuatNorm(Quat{W: math.Cos(half), X: w.X * s, Y: w.Y * s, Z: w.Z * s})
}

// QuatToRotVec 四元数转旋转向量，小角度安全，结果模长 ∈ [0, π]。
func QuatToRotVec(q Quat) Vec3 {
	q = QuatNorm(q)
	if q.W < 0 { // 取双覆盖中的短路径
		q = Quat{-q.W, -q.X, -q.Y, -q.Z}
	}
	sinHalf := math.Sqrt(q.X*q.X + q.Y*q.Y + q.Z*q.Z)
	if sinHalf < 1e-8 {
		return Vec3{2 * q.X, 2 * q.Y, 2 * q.Z}
	}
	angle := 2 * math.Atan2(sinHalf, q.W)
	s := angle / sinHalf
	return Vec3{q.X * s, q.Y * s, q.Z * s}
}

// QuatSlerp 球面插值，t∈[0,1]，处理近零夹角与双覆盖。
func QuatSlerp(a, b Quat, t float64) Quat {
	a = QuatNorm(a)
	b = QuatNorm(b)
	dot := a.W*b.W + a.X*b.X + a.Y*b.Y + a.Z*b.Z
	if dot < 0 { // 最短路径
		b = Quat{-b.W, -b.X, -b.Y, -b.Z}
		dot = -dot
	}
	if dot > 0.9995 { // 近零夹角退化为线性插值
		return QuatNorm(Quat{
			W: a.W + t*(b.W-a.W),
			X: a.X + t*(b.X-a.X),
			Y: a.Y + t*(b.Y-a.Y),
			Z: a.Z + t*(b.Z-a.Z),
		})
	}
	theta := math.Acos(dot)
	sinT := math.Sin(theta)
	wa := math.Sin((1-t)*theta) / sinT
	wb := math.Sin(t*theta) / sinT
	return QuatNorm(Quat{
		W: wa*a.W + wb*b.W,
		X: wa*a.X + wb*b.X,
		Y: wa*a.Y + wb*b.Y,
		Z: wa*a.Z + wb*b.Z,
	})
}

// Vec3 基本运算。
func vecAdd(a, b Vec3) Vec3           { return Vec3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func vecSub(a, b Vec3) Vec3           { return Vec3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }
func vecScale(a Vec3, s float64) Vec3 { return Vec3{a.X * s, a.Y * s, a.Z * s} }
func vecDot(a, b Vec3) float64        { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func vecNorm(a Vec3) float64          { return math.Sqrt(vecDot(a, a)) }
func vecCross(a, b Vec3) Vec3 {
	return Vec3{
		X: a.Y*b.Z - a.Z*b.Y,
		Y: a.Z*b.X - a.X*b.Z,
		Z: a.X*b.Y - a.Y*b.X,
	}
}
