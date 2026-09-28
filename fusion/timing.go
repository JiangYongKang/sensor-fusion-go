package fusion

import "math"

// timingEvidence 从一次创新评估中提取的时钟时序证据。
//
// 对线性化观测残差 r = h(t_phys)−h(T) ≈ −ḣ·(T−t_phys)，
// 标量最优时序修正 τ* = t_phys−T 由加权一维投影给出：
//
//	τ* = (rᵀW ḣ)/(ḣᵀW ḣ),  σ²τ = 1/(ḣᵀW ḣ)
//
// 全部运算为确定性浮点运算，相同输入逐位一致。
type timingEvidence struct {
	tau     float64 // 推断的 (物理时刻 − 当前状态时刻)，秒
	measVar float64 // τ 的测量方差，秒²
	valid   bool
}

// scalarTiming 通用加权标量投影。res/rates/weights 等长。
// 运动能量不足（分母过小）时 invalid。
func scalarTiming(res, rates, weights []float64, minEnergy float64) timingEvidence {
	var num, den float64
	for i := range res {
		w := weights[i]
		num += w * res[i] * rates[i]
		den += w * rates[i] * rates[i]
	}
	if den <= minEnergy || math.IsNaN(den) {
		return timingEvidence{valid: false}
	}
	return timingEvidence{tau: num / den, measVar: 1.0 / den, valid: true}
}

// poseTimingEvidence 绝对位姿创新的时序证据：
// 位置块用世界系线速度，姿态块用世界系角速度；仅纳入运动足够的块。
func poseTimingEvidence(iv innovation, velWorld, angWorld Vec3, minLin, minAng float64) timingEvidence {
	var res, rates, weights []float64
	if n := vecNorm(velWorld); n >= minLin {
		res = append(res, iv.res[0], iv.res[1], iv.res[2])
		rates = append(rates, velWorld.X, velWorld.Y, velWorld.Z)
		weights = append(weights, 1/iv.rDiag[0], 1/iv.rDiag[1], 1/iv.rDiag[2])
	}
	if n := vecNorm(angWorld); n >= minAng {
		res = append(res, iv.res[3], iv.res[4], iv.res[5])
		rates = append(rates, angWorld.X, angWorld.Y, angWorld.Z)
		weights = append(weights, 1/iv.rDiag[3], 1/iv.rDiag[4], 1/iv.rDiag[5])
	}
	if len(res) == 0 {
		return timingEvidence{valid: false}
	}
	// 能量门限：各块速率模长已单独判定，这里 den 必然为正；给一个数值下界。
	return scalarTiming(res, rates, weights, 1e-12)
}

// odomTimingEvidence 里程计（机体系速度）创新的时序证据。
// 世界系线加速度 accWorld（已含重力补偿）直接转到机体系作为速率，
// 避免用含误差的姿态估计离心项；运动能量不足时 invalid。
func odomTimingEvidence(iv innovation, q Quat, accWorld Vec3, minAccel float64) timingEvidence {
	rateB := QuatRotate(QuatConj(q), accWorld)
	if vecNorm(rateB) < minAccel {
		return timingEvidence{valid: false}
	}
	res := iv.res
	rates := []float64{rateB.X, rateB.Y, rateB.Z}
	weights := []float64{1 / iv.rDiag[0], 1 / iv.rDiag[1], 1 / iv.rDiag[2]}
	return scalarTiming(res, rates, weights, 1e-12)
}
