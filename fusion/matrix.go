package fusion

import "math"

// Mat 是行优先存储的小规模稠密矩阵（最大 9x9），
// 用于误差状态协方差与卡尔曼更新，所有操作确定性执行。
type Mat struct {
	N    int
	Data [81]float64
}

// NewMat 创建 n×n 零矩阵。n 超界时钳制到 [1,9]。
func NewMat(n int) Mat {
	if n < 1 {
		n = 1
	}
	if n > 9 {
		n = 9
	}
	return Mat{N: n}
}

// MatDiag 由对角元构造矩阵。
func MatDiag(d []float64) Mat {
	m := NewMat(len(d))
	for i, v := range d {
		m.Data[i*m.N+i] = v
	}
	return m
}

func (m Mat) At(i, j int) float64 { return m.Data[i*m.N+j] }

func (m *Mat) Set(i, j int, v float64) { m.Data[i*m.N+j] = v }

// Identity 单位矩阵。
func Identity(n int) Mat {
	m := NewMat(n)
	for i := 0; i < n; i++ {
		m.Data[i*n+i] = 1
	}
	return m
}

// Add 返回 a+b（维度不一致时按较小维度，防御性处理）。
func (m Mat) Add(o Mat) Mat {
	n := m.N
	if o.N < n {
		n = o.N
	}
	r := NewMat(n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			r.Data[i*n+j] = m.Data[i*m.N+j] + o.Data[i*o.N+j]
		}
	}
	return r
}

// Mul 返回 a·b。
func (m Mat) Mul(o Mat) Mat {
	r := NewMat(m.N)
	for i := 0; i < m.N; i++ {
		for k := 0; k < m.N; k++ {
			a := m.Data[i*m.N+k]
			if a == 0 {
				continue
			}
			for j := 0; j < m.N; j++ {
				r.Data[i*m.N+j] += a * o.Data[k*m.N+j]
			}
		}
	}
	return r
}

// MulVec 返回 M·v。
func (m Mat) MulVec(v []float64) []float64 {
	r := make([]float64, m.N)
	for i := 0; i < m.N; i++ {
		var s float64
		for j := 0; j < m.N && j < len(v); j++ {
			s += m.Data[i*m.N+j] * v[j]
		}
		r[i] = s
	}
	return r
}

// Transpose 返回转置。
func (m Mat) Transpose() Mat {
	r := NewMat(m.N)
	for i := 0; i < m.N; i++ {
		for j := 0; j < m.N; j++ {
			r.Data[j*m.N+i] = m.Data[i*m.N+j]
		}
	}
	return r
}

// Symmetrize 原地对称化 M ← (M+Mᵀ)/2，抑制浮点非对称漂移。
func (m *Mat) Symmetrize() {
	for i := 0; i < m.N; i++ {
		for j := i + 1; j < m.N; j++ {
			avg := (m.Data[i*m.N+j] + m.Data[j*m.N+i]) / 2
			m.Data[i*m.N+j] = avg
			m.Data[j*m.N+i] = avg
		}
	}
}

// Cholesky 下三角分解 A = L·Lᵀ，非正定时返回 false。
func (m Mat) Cholesky() (Mat, bool) {
	L := NewMat(m.N)
	for i := 0; i < m.N; i++ {
		for j := 0; j <= i; j++ {
			s := m.Data[i*m.N+j]
			for k := 0; k < j; k++ {
				s -= L.Data[i*m.N+k] * L.Data[j*m.N+k]
			}
			if i == j {
				if s <= 0 || math.IsNaN(s) {
					return Mat{}, false
				}
				L.Data[i*m.N+i] = math.Sqrt(s)
			} else {
				L.Data[i*m.N+j] = s / L.Data[j*m.N+j]
			}
		}
	}
	return L, true
}

// SolveLower 解 L x = b（前代）。
func SolveLower(L Mat, b []float64) []float64 {
	n := L.N
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		s := b[i]
		for k := 0; k < i; k++ {
			s -= L.Data[i*n+k] * x[k]
		}
		x[i] = s / L.Data[i*n+i]
	}
	return x
}

// SolveUpper 解 Lᵀ x = b（回代）。
func SolveUpper(L Mat, b []float64) []float64 {
	n := L.N
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := b[i]
		for k := i + 1; k < n; k++ {
			s -= L.Data[k*n+i] * x[k]
		}
		x[i] = s / L.Data[i*n+i]
	}
	return x
}

// IsPositiveDefinite 通过 Cholesky 判定正定性。
func (m Mat) IsPositiveDefinite() bool {
	_, ok := m.Cholesky()
	return ok
}

// ClampDiagFloor 将协方差对角元钳制到下界，保证正定性下限。
func (m *Mat) ClampDiagFloor(floor float64) {
	for i := 0; i < m.N; i++ {
		if m.Data[i*m.N+i] < floor {
			m.Data[i*m.N+i] = floor
		}
	}
}
