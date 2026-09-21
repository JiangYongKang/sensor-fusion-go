// Package linalg provides a small, self-contained set of fixed-size
// linear-algebra primitives used by the estimation engine. No third-party
// dependencies are used so that numerical behaviour is fully reproducible.
package linalg

import (
	"fmt"
	"math"
	"strings"
)

// Vec is a dense column vector.
type Vec struct{ X []float64 }

// Mat is a row-major dense matrix with R rows and C columns.
type Mat struct {
	R, C int
	X    []float64
}

func NewVec(n int) Vec { return Vec{X: make([]float64, n)} }

func VecOf(x ...float64) Vec { return Vec{X: append([]float64(nil), x...)} }

func (v Vec) At(i int) float64     { return v.X[i] }
func (v Vec) Set(i int, x float64) { v.X[i] = x }
func (v Vec) Len() int             { return len(v.X) }

func (v Vec) Clone() Vec { return Vec{X: append([]float64(nil), v.X...)} }

func (v Vec) Slice(a, b int) Vec {
	x := append([]float64(nil), v.X[a:b]...)
	return Vec{X: x}
}

func checkSame(a, b Vec) {
	if len(a.X) != len(b.X) {
		panic(fmt.Sprintf("linalg: vector length mismatch %d vs %d", len(a.X), len(b.X)))
	}
}

func (v Vec) Add(o Vec) Vec {
	checkSame(v, o)
	r := NewVec(len(v.X))
	for i := range v.X {
		r.X[i] = v.X[i] + o.X[i]
	}
	return r
}

func (v Vec) Sub(o Vec) Vec {
	checkSame(v, o)
	r := NewVec(len(v.X))
	for i := range v.X {
		r.X[i] = v.X[i] - o.X[i]
	}
	return r
}

func (v Vec) Scale(s float64) Vec {
	r := NewVec(len(v.X))
	for i := range v.X {
		r.X[i] = v.X[i] * s
	}
	return r
}

func (v Vec) Dot(o Vec) float64 {
	checkSame(v, o)
	var s float64
	for i := range v.X {
		s += v.X[i] * o.X[i]
	}
	return s
}

func (v Vec) Norm2() float64 { return math.Sqrt(v.Dot(v)) }

func (v Vec) String() string {
	var b strings.Builder
	b.WriteByte('[')
	for i, x := range v.X {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%.9g", x)
	}
	b.WriteByte(']')
	return b.String()
}

func NewMat(r, c int) Mat { return Mat{R: r, C: c, X: make([]float64, r*c)} }

func Zeros(r, c int) Mat { return NewMat(r, c) }

func MatFrom(r, c int, x []float64) Mat {
	if len(x) != r*c {
		panic("linalg: MatFrom size mismatch")
	}
	return Mat{R: r, C: c, X: append([]float64(nil), x...)}
}

func Identity(n int) Mat {
	m := NewMat(n, n)
	for i := 0; i < n; i++ {
		m.X[i*n+i] = 1
	}
	return m
}

func (m Mat) idx(i, j int) int {
	if i < 0 || i >= m.R || j < 0 || j >= m.C {
		panic(fmt.Sprintf("linalg: index (%d,%d) out of %dx%d", i, j, m.R, m.C))
	}
	return i*m.C + j
}

func (m Mat) At(i, j int) float64     { return m.X[m.idx(i, j)] }
func (m Mat) Set(i, j int, v float64) { m.X[m.idx(i, j)] = v }

func (m Mat) Clone() Mat {
	return Mat{R: m.R, C: m.C, X: append([]float64(nil), m.X...)}
}

func (m Mat) Transpose() Mat {
	r := NewMat(m.C, m.R)
	for i := 0; i < m.R; i++ {
		for j := 0; j < m.C; j++ {
			r.X[j*r.C+i] = m.X[i*m.C+j]
		}
	}
	return r
}

func (m Mat) Mul(o Mat) Mat {
	if m.C != o.R {
		panic(fmt.Sprintf("linalg: matmul %dx%d * %dx%d", m.R, m.C, o.R, o.C))
	}
	r := NewMat(m.R, o.C)
	for i := 0; i < m.R; i++ {
		for k := 0; k < m.C; k++ {
			a := m.X[i*m.C+k]
			if a == 0 {
				continue
			}
			for j := 0; j < o.C; j++ {
				r.X[i*r.C+j] += a * o.X[k*o.C+j]
			}
		}
	}
	return r
}

func (m Mat) MulVec(v Vec) Vec {
	if m.C != len(v.X) {
		panic("linalg: matrix-vector size mismatch")
	}
	r := NewVec(m.R)
	for i := 0; i < m.R; i++ {
		var s float64
		base := i * m.C
		for j := 0; j < m.C; j++ {
			s += m.X[base+j] * v.X[j]
		}
		r.X[i] = s
	}
	return r
}

func sameShape(a, b Mat) {
	if a.R != b.R || a.C != b.C {
		panic("linalg: matrix shape mismatch")
	}
}

func (m Mat) Add(o Mat) Mat {
	sameShape(m, o)
	r := NewMat(m.R, m.C)
	for i := range m.X {
		r.X[i] = m.X[i] + o.X[i]
	}
	return r
}

func (m Mat) Sub(o Mat) Mat {
	sameShape(m, o)
	r := NewMat(m.R, m.C)
	for i := range m.X {
		r.X[i] = m.X[i] - o.X[i]
	}
	return r
}

func (m Mat) Scale(s float64) Mat {
	r := NewMat(m.R, m.C)
	for i := range m.X {
		r.X[i] = m.X[i] * s
	}
	return r
}

// Symmetrize returns 0.5*(m + m^T). Panics for non-square matrices.
func (m Mat) Symmetrize() Mat {
	if m.R != m.C {
		panic("linalg: Symmetrize on non-square matrix")
	}
	r := NewMat(m.R, m.R)
	for i := 0; i < m.R; i++ {
		for j := i; j < m.R; j++ {
			v := 0.5 * (m.X[i*m.R+j] + m.X[j*m.R+i])
			r.X[i*m.R+j] = v
			r.X[j*m.R+i] = v
		}
	}
	return r
}

// Cholesky computes L such that m = L*L^T for a symmetric positive-definite
// matrix. ok is false if the matrix is not positive definite.
func (m Mat) Cholesky() (Mat, bool) {
	if m.R != m.C {
		return Mat{}, false
	}
	n := m.R
	L := NewMat(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			var s float64
			for k := 0; k < j; k++ {
				s += L.X[i*n+k] * L.X[j*n+k]
			}
			if i == j {
				d := 0.5*(m.X[i*n+i]+m.X[i*n+i]) - s
				if !(d > 0) || math.IsNaN(d) || math.IsInf(d, 0) {
					return Mat{}, false
				}
				L.X[i*n+j] = math.Sqrt(d)
			} else {
				L.X[i*n+j] = (m.X[i*n+j] - s) / L.X[j*n+j]
			}
		}
	}
	return L, true
}

// SolveSPD solves m*x = b where m is symmetric positive definite, using a
// Cholesky factorisation.
func (m Mat) SolveSPD(b Vec) (Vec, bool) {
	L, ok := m.Cholesky()
	if !ok {
		return Vec{}, false
	}
	y := SolveLowerTri(L, b)
	x := SolveUpperTri(L.Transpose(), y)
	return x, true
}

// SolveLowerTri solves L*x = b by forward substitution.
func SolveLowerTri(L Mat, b Vec) Vec {
	n := L.R
	x := NewVec(n)
	for i := 0; i < n; i++ {
		var s float64
		for j := 0; j < i; j++ {
			s += L.X[i*n+j] * x.X[j]
		}
		d := L.X[i*n+i]
		if d == 0 {
			return NewVec(n)
		}
		x.X[i] = (b.X[i] - s) / d
	}
	return x
}

// SolveUpperTri solves U*x = b by back substitution.
func SolveUpperTri(U Mat, b Vec) Vec {
	n := U.R
	x := NewVec(n)
	for i := n - 1; i >= 0; i-- {
		var s float64
		for j := i + 1; j < n; j++ {
			s += U.X[i*n+j] * x.X[j]
		}
		d := U.X[i*n+i]
		if d == 0 {
			return NewVec(n)
		}
		x.X[i] = (b.X[i] - s) / d
	}
	return x
}

// Eigen3 computes the eigenvalues and orthonormal eigenvectors of a
// symmetric 3x3 matrix using the analytic trigonometric (Cardano) method.
// Eigenvalues are returned in descending order; column k of vecs is the
// eigenvector of vals[k]. ok is false for non-finite input.
func (m Mat) Eigen3() (vals [3]float64, vecs [3][3]float64, ok bool) {
	if m.R != 3 || m.C != 3 {
		return [3]float64{}, [3][3]float64{}, false
	}
	a := m.Symmetrize()
	for _, x := range a.X {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return [3]float64{}, [3][3]float64{}, false
		}
	}
	p1 := a.X[1]*a.X[1] + a.X[2]*a.X[2] + a.X[5]*a.X[5]
	var eig [3]float64
	if p1 <= 1e-30 {
		// Matrix is diagonal.
		eig = [3]float64{a.X[0], a.X[4], a.X[8]}
	} else {
		tr := a.X[0] + a.X[4] + a.X[8]
		q := tr / 3
		p2 := (a.X[0]-q)*(a.X[0]-q) + (a.X[4]-q)*(a.X[4]-q) + (a.X[8]-q)*(a.X[8]-q) + 2*p1
		p := math.Sqrt(p2 / 6)
		if p == 0 {
			eig = [3]float64{q, q, q}
		} else {
			det := (a.X[0]-q)*((a.X[4]-q)*(a.X[8]-q)-a.X[5]*a.X[7]) -
				a.X[1]*(a.X[3]*(a.X[8]-q)-a.X[5]*a.X[6]) +
				a.X[2]*(a.X[3]*a.X[7]-(a.X[4]-q)*a.X[6])
			r := det / (2 * p * p * p)
			if r > 1 {
				r = 1
			}
			if r < -1 {
				r = -1
			}
			phi := math.Acos(r) / 3
			eig[0] = q + 2*p*math.Cos(phi)
			eig[2] = q + 2*p*math.Cos(phi+2*math.Pi/3)
			eig[1] = 3*q - eig[0] - eig[2]
		}
	}
	// Sort descending (simple exchange network).
	idx := [3]int{0, 1, 2}
	if eig[idx[0]] < eig[idx[1]] {
		idx[0], idx[1] = idx[1], idx[0]
	}
	if eig[idx[1]] < eig[idx[2]] {
		idx[1], idx[2] = idx[2], idx[1]
	}
	if eig[idx[0]] < eig[idx[1]] {
		idx[0], idx[1] = idx[1], idx[0]
	}
	for k, ii := range idx {
		vals[k] = eig[ii]
	}
	// Eigenvectors via null-space of (m - lambda I), with fallbacks.
	for k := 0; k < 3; k++ {
		B := NewMat(3, 3)
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				B.X[i*3+j] = a.X[i*3+j]
			}
			B.X[i*3+i] -= vals[k]
		}
		v := nullVector3(B)
		vecs[0][k], vecs[1][k], vecs[2][k] = v[0], v[1], v[2]
	}
	// Deterministic sign convention: largest-magnitude component positive.
	for k := 0; k < 3; k++ {
		mi := 0
		for i := 1; i < 3; i++ {
			if math.Abs(vecs[i][k]) > math.Abs(vecs[mi][k]) {
				mi = i
			}
		}
		if vecs[mi][k] < 0 {
			for i := 0; i < 3; i++ {
				vecs[i][k] = -vecs[i][k]
			}
		}
	}
	return vals, vecs, true
}

// nullVector3 returns a unit vector in the null space of a rank-deficient
// 3x3 matrix, choosing the most independent of three candidate cross
// products for numerical robustness.
func nullVector3(B Mat) [3]float64 {
	row := func(i int) [3]float64 {
		return [3]float64{B.X[i*3], B.X[i*3+1], B.X[i*3+2]}
	}
	r0, r1, r2 := row(0), row(1), row(2)
	c01 := cross3(r0, r1)
	c02 := cross3(r0, r2)
	c12 := cross3(r1, r2)
	cands := [][3]float64{c01, c02, c12}
	best := c01
	bestN := norm3(c01)
	for _, c := range cands[1:] {
		if n := norm3(c); n > bestN {
			best, bestN = c, n
		}
	}
	if bestN <= 1e-18 {
		// (Near-)repeated eigenvalue: use a fixed axis least aligned with
		// the diagonal direction.
		best = [3]float64{1, 0, 0}
	}
	n := norm3(best)
	return [3]float64{best[0] / n, best[1] / n, best[2] / n}
}

func cross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func norm3(a [3]float64) float64 {
	return math.Sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
}

func (m Mat) String() string {
	var b strings.Builder
	for i := 0; i < m.R; i++ {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteByte('[')
		for j := 0; j < m.C; j++ {
			if j > 0 {
				b.WriteByte(' ')
			}
			fmt.Fprintf(&b, "%.9g", m.X[i*m.C+j])
		}
		b.WriteByte(']')
	}
	return b.String()
}
