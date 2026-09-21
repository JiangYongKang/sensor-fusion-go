package fusion

import (
	"math"
	"time"
)

// State 名义状态：位置、速度、姿态（世界系，ENU 约定）。
type State struct {
	Position Vec3
	Velocity Vec3
	Rotation Quat
	Stamp    time.Time // 状态对应时间
}

// ErrorStateDim 误差状态维度 [δp, δv, δθ]。
const ErrorStateDim = 9

// Estimator 误差状态卡尔曼估计器（ESKF）。
// 旋转以单位四元数承载，每步预测后归一化；
// 协方差经 Joseph 形式更新并每次对称化，保证正定性。
type Estimator struct {
	Nominal State
	Cov     Mat // 9x9 误差状态协方差
	gravity Vec3
}

// NewEstimator 以给定初始位姿构造估计器。
func NewEstimator(init State, initCovDiag [9]float64) *Estimator {
	init.Rotation = QuatNorm(init.Rotation)
	cov := MatDiag(initCovDiag[:])
	cov.ClampDiagFloor(1e-12)
	return &Estimator{
		Nominal: init,
		Cov:     cov,
		gravity: Vec3{Z: -9.81},
	}
}

// Predict 用 IMU 观测将状态推进 dt 秒。
// 名义状态：p ← p + v·dt + ½(R·a+g)dt²；v ← v + (R·a+g)dt；q ← q⊗exp(ω·dt)。
// 协方差：P ← F·P·Fᵀ（过程噪声由 AddProcessNoise 另行加入）。
func (e *Estimator) Predict(imu Observation, dt float64) {
	if dt <= 0 {
		return
	}
	n := &e.Nominal
	accW := vecAdd(QuatRotate(n.Rotation, imu.Accel), e.gravity)

	n.Position = vecAdd(n.Position, vecAdd(vecScale(n.Velocity, dt), vecScale(accW, 0.5*dt*dt)))
	n.Velocity = vecAdd(n.Velocity, vecScale(accW, dt))
	n.Rotation = QuatNorm(QuatMul(n.Rotation, QuatFromRotVec(vecScale(imu.Gyro, dt))))
	n.Stamp = n.Stamp.Add(time.Duration(dt * float64(time.Second)))

	// 误差状态转移 F = I + Fc·dt
	F := Identity(ErrorStateDim)
	skewA := skew(accW)
	for i := 0; i < 3; i++ {
		F.Set(i, 3+i, dt) // δp ← δv
		for j := 0; j < 3; j++ {
			F.Set(3+i, 6+j, -skewA.At(i, j)*dt) // δv ← δθ
		}
	}
	// δθ ← δθ：I − [ω·dt]×
	skewW := skew(vecScale(imu.Gyro, dt))
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			v := -skewW.At(i, j)
			if i == j {
				v += 1
			}
			F.Set(6+i, 6+j, v)
		}
	}

	e.Cov = F.Mul(e.Cov).Mul(F.Transpose())
	e.Cov.Symmetrize()
	e.Cov.ClampDiagFloor(1e-12)
}

// AddProcessNoise 将过程噪声对角按 dt 缩放加入协方差。
func (e *Estimator) AddProcessNoise(diag [9]float64, dt float64) {
	for i := 0; i < ErrorStateDim; i++ {
		e.Cov.Set(i, i, e.Cov.At(i, i)+diag[i]*dt)
	}
}

// update 通用线性观测更新。
// res: m 维残差；H: m×9 行优先；rDiag: m 维观测噪声对角。
// 返回马氏平方与是否通过门限。Joseph 形式保证协方差正定性。
func (e *Estimator) update(res []float64, H []float64, rDiag []float64, gateChi2 float64) (float64, bool) {
	m := len(res)
	// S = H·P·Hᵀ + R（m×m）
	S := NewMat(m)
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			var s float64
			for a := 0; a < ErrorStateDim; a++ {
				hia := H[i*ErrorStateDim+a]
				if hia == 0 {
					continue
				}
				for b := 0; b < ErrorStateDim; b++ {
					s += hia * e.Cov.At(a, b) * H[j*ErrorStateDim+b]
				}
			}
			if i == j {
				s += rDiag[i]
			}
			S.Set(i, j, s)
		}
	}
	L, ok := S.Cholesky()
	if !ok {
		return math.Inf(1), false // 数值异常，拒绝且不污染状态
	}
	// 马氏平方 d² = resᵀ·S⁻¹·res
	y := SolveLower(L, res)
	var d2 float64
	for _, v := range y {
		d2 += v * v
	}
	if d2 > gateChi2 {
		return d2, false
	}
	// K = P·Hᵀ·S⁻¹（9×m）。转置求解：对 K 的每一行 r，
	// 解 S·x = (P·Hᵀ) 的第 r 行（m 维），x 即 K 的第 r 行。
	K := make([]float64, ErrorStateDim*m) // 行优先 9×m
	for r := 0; r < ErrorStateDim; r++ {
		b := make([]float64, m)
		for c := 0; c < m; c++ {
			var s float64
			for a := 0; a < ErrorStateDim; a++ {
				s += e.Cov.At(r, a) * H[c*ErrorStateDim+a]
			}
			b[c] = s
		}
		sol := SolveUpper(L, SolveLower(L, b))
		for c := 0; c < m; c++ {
			K[r*m+c] = sol[c]
		}
	}
	// dx = K·res
	dx := make([]float64, ErrorStateDim)
	for r := 0; r < ErrorStateDim; r++ {
		var s float64
		for c := 0; c < m; c++ {
			s += K[r*m+c] * res[c]
		}
		dx[r] = s
	}
	// Joseph 形式：P ← (I−K·H)P(I−K·H)ᵀ + K·R·Kᵀ
	IKH := Identity(ErrorStateDim)
	for r := 0; r < ErrorStateDim; r++ {
		for c := 0; c < ErrorStateDim; c++ {
			var s float64
			for k := 0; k < m; k++ {
				s += K[r*m+k] * H[k*ErrorStateDim+c]
			}
			IKH.Set(r, c, IKH.At(r, c)-s)
		}
	}
	KRK := NewMat(ErrorStateDim)
	for r := 0; r < ErrorStateDim; r++ {
		for c := 0; c < ErrorStateDim; c++ {
			var s float64
			for k := 0; k < m; k++ {
				s += K[r*m+k] * rDiag[k] * K[c*m+k]
			}
			KRK.Set(r, c, s)
		}
	}
	e.Cov = IKH.Mul(e.Cov).Mul(IKH.Transpose()).Add(KRK)
	e.Cov.Symmetrize()
	e.Cov.ClampDiagFloor(1e-12)

	e.injectError(dx)
	return d2, true
}

// UpdateOdometry 里程计速度观测（机体系线速度），返回马氏平方与是否被接受。
func (e *Estimator) UpdateOdometry(o Observation, gateChi2 float64) (float64, bool) {
	// 预测观测：机体系速度 = Rᵀ·v_world
	pred := QuatRotate(QuatConj(e.Nominal.Rotation), e.Nominal.Velocity)
	res := []float64{o.LinVel.X - pred.X, o.LinVel.Y - pred.Y, o.LinVel.Z - pred.Z}
	// H（3×9）：δv 部分 Rᵀ，δθ 部分 [Rᵀ·v]×（由 h'=(I−[δθ]×)Rᵀv 推得）
	Rt := quatToRotMat(QuatConj(e.Nominal.Rotation))
	skewPred := skew(pred)
	H := make([]float64, 3*ErrorStateDim)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			H[i*ErrorStateDim+3+j] = Rt.At(i, j)
			H[i*ErrorStateDim+6+j] = skewPred.At(i, j)
		}
	}
	rDiag := []float64{o.CovDiag[0], o.CovDiag[1], o.CovDiag[2]}
	return e.update(res, H, rDiag, gateChi2)
}

// UpdateAbsolutePose 绝对位姿观测更新。
func (e *Estimator) UpdateAbsolutePose(o Observation, gateChi2 float64) (float64, bool) {
	// 位置残差
	resP := vecSub(o.Pose.Position, e.Nominal.Position)
	// 姿态残差：δθ = log(q̂⁻¹ ⊗ q_obs)
	qErr := QuatMul(QuatConj(e.Nominal.Rotation), o.Pose.Rotation)
	resR := QuatToRotVec(qErr)
	res := []float64{resP.X, resP.Y, resP.Z, resR.X, resR.Y, resR.Z}
	// H（6×9）：δp→I，δθ→I（6..8 列）
	H := make([]float64, 6*ErrorStateDim)
	for i := 0; i < 3; i++ {
		H[i*ErrorStateDim+i] = 1
		H[(3+i)*ErrorStateDim+6+i] = 1
	}
	rDiag := []float64{
		o.CovDiag[0], o.CovDiag[1], o.CovDiag[2],
		o.CovDiag[3], o.CovDiag[4], o.CovDiag[5],
	}
	return e.update(res, H, rDiag, gateChi2)
}

// Snapshot 返回状态深拷贝快照（供并发读取）。
func (e *Estimator) Snapshot() (State, Mat) {
	return e.Nominal, e.Cov
}

// injectError 将误差状态注入名义状态并复位。
func (e *Estimator) injectError(dx []float64) {
	n := &e.Nominal
	n.Position = vecAdd(n.Position, Vec3{dx[0], dx[1], dx[2]})
	n.Velocity = vecAdd(n.Velocity, Vec3{dx[3], dx[4], dx[5]})
	n.Rotation = QuatNorm(QuatMul(n.Rotation, QuatFromRotVec(Vec3{dx[6], dx[7], dx[8]})))
}

// quatToRotMat 四元数转 3x3 旋转矩阵。
func quatToRotMat(q Quat) Mat {
	q = QuatNorm(q)
	m := NewMat(3)
	w, x, y, z := q.W, q.X, q.Y, q.Z
	m.Set(0, 0, 1-2*(y*y+z*z))
	m.Set(0, 1, 2*(x*y-w*z))
	m.Set(0, 2, 2*(x*z+w*y))
	m.Set(1, 0, 2*(x*y+w*z))
	m.Set(1, 1, 1-2*(x*x+z*z))
	m.Set(1, 2, 2*(y*z-w*x))
	m.Set(2, 0, 2*(x*z-w*y))
	m.Set(2, 1, 2*(y*z+w*x))
	m.Set(2, 2, 1-2*(x*x+y*y))
	return m
}

// skew 向量反对称矩阵。
func skew(v Vec3) Mat {
	m := NewMat(3)
	m.Set(0, 1, -v.Z)
	m.Set(0, 2, v.Y)
	m.Set(1, 0, v.Z)
	m.Set(1, 2, -v.X)
	m.Set(2, 0, -v.Y)
	m.Set(2, 1, v.X)
	return m
}

// PredictConstantVelocity 无 IMU 时的恒定速度模型推进：
// p ← p + v·dt，速度与姿态保持不变，协方差按 δp←δv 耦合传播。
func (e *Estimator) PredictConstantVelocity(dt float64) {
	if dt <= 0 {
		return
	}
	n := &e.Nominal
	n.Position = vecAdd(n.Position, vecScale(n.Velocity, dt))
	n.Stamp = n.Stamp.Add(time.Duration(dt * float64(time.Second)))

	F := Identity(ErrorStateDim)
	for i := 0; i < 3; i++ {
		F.Set(i, 3+i, dt)
	}
	e.Cov = F.Mul(e.Cov).Mul(F.Transpose())
	e.Cov.Symmetrize()
	e.Cov.ClampDiagFloor(1e-12)
}
