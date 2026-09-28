package fusion

import (
	"math"
	"time"
)

// clockPhase 描述一路传感器时钟模型所处的阶段。
type clockPhase int

const (
	phaseReference   clockPhase = iota // 参考时钟（IMU）：恒等映射，恒可信
	phaseCalibrating                   // 冷启动样本不足，换算不可信
	phaseProvisional                   // 恒等先验，持续积累证据；映射为精确恒等
	phaseActive                        // 估计模型生效，换算可信
	phaseOutOfRange                    // 漂移越界（粘滞），换算不可信
)

func (p clockPhase) String() string {
	switch p {
	case phaseReference:
		return "reference"
	case phaseCalibrating:
		return "calibrating"
	case phaseProvisional:
		return "provisional"
	case phaseActive:
		return "active"
	case phaseOutOfRange:
		return "out_of_range"
	}
	return "unknown"
}

// clockSample 一个时钟观测样本。
type clockSample struct {
	src     float64 // 传感器时钟读数（相对该路锚点，秒）
	uni     float64 // 推断的统一时间（相对同一锚点，秒）
	measVar float64 // 该样本的测量方差（秒²）
}

// clockModel 单路时钟的在线仿射模型：
//
//	unified = anchor + b + s·(stamp-anchor)
//
// b、s 由带遗忘因子的递归最小二乘在线辨识。阶段机：
//
//	calibrating →（样本充足）→ provisional（偏移不显著）或 active（显著）
//	provisional →（偏移连续显著）→ active
//	任意阶段发现 |b|、偏移范围或 |s-1| 越界 → out_of_range（粘滞）
//
// provisional/reference 始终走精确恒等映射，因此无漂移输入的全部
// 时间判定（排序、dt、陈旧窗口）与旧版本逐位一致。
type clockModel struct {
	kind    SensorKind
	phase   clockPhase
	anchor  time.Time
	b       float64 // 截距（秒）
	s       float64 // 斜率（1=无频偏）
	n       int
	confirm int
	maxSrc  float64

	// 差分频偏估计（冷启动稳健）：仅由相邻同类观测的读数间隔与
	// 物理间隔之比更新，不依赖初始位姿或长程积分。
	lastSrc    float64
	lastUni    float64
	hasPair    bool
	offCand    float64 // 恒等先验期累计偏移证据（不改变 theta）
	candMean   float64
	candM2     float64
	candCnt    int
	lastSign   float64
	skewEMA    float64
	skewN      int
	lastRatios []float64 // 诊断：近期区间比
}

// rlsState 2 状态 RLS 辨识器（theta=[b,s]，2x2 对称协方差）。
type rlsState struct {
	theta [2]float64
	P     [4]float64
}

func newRLS(initBStd, initSStd float64) rlsState {
	return rlsState{
		theta: [2]float64{0, 1},
		P:     [4]float64{initBStd * initBStd, 0, 0, initSStd * initSStd},
	}
}

// observe 执行一步 RLS：y = h·theta + v，h=[1,x]，v 方差 measVar。
// 过程噪声按采样间隔 dt 注入，遗忘因子 lambda∈(0,1] 跟踪缓变漂移。
func (r *rlsState) observe(x, y, measVar, dt, lambda, qB, qS float64) {
	r.P[0] += qB * qB * dt
	r.P[3] += qS * qS * dt
	h0, h1 := 1.0, x
	ph0 := r.P[0]*h0 + r.P[1]*h1
	ph1 := r.P[2]*h0 + r.P[3]*h1
	denom := h0*ph0 + h1*ph1 + measVar
	if denom <= 0 || math.IsNaN(denom) || math.IsInf(denom, 0) {
		return
	}
	k0, k1 := ph0/denom, ph1/denom
	innov := y - (r.theta[0] + r.theta[1]*x)
	r.theta[0] += k0 * innov
	r.theta[1] += k1 * innov
	// P ← (I−K·h)·P / λ，随后对称化（确定性，无随机分支）。
	p00, p01 := r.P[0], r.P[1]
	p10, p11 := r.P[2], r.P[3]
	kh00, kh01 := k0*h0, k0*h1
	kh10, kh11 := k1*h0, k1*h1
	np00 := (1-kh00)*p00 - kh01*p10
	np01 := (1-kh00)*p01 - kh01*p11
	np10 := -kh10*p00 + (1-kh11)*p10
	np11 := -kh10*p01 + (1-kh11)*p11
	r.P[0] = np00 / lambda
	r.P[1] = np01 / lambda
	r.P[2] = np10 / lambda
	r.P[3] = np11 / lambda
	// 强制对称并保留下界，抑制长期递推的非对称漂移。
	off := 0.5 * (r.P[1] + r.P[2])
	r.P[1], r.P[2] = off, off
	const pFloor = 1e-12
	if r.P[0] < pFloor {
		r.P[0] = pFloor
	}
	if r.P[3] < pFloor {
		r.P[3] = pFloor
	}
}

// observeOffset 仅更新截距 b（斜率固定为 1）。模型 y = b + x，
// 残差 (y−x) 即偏移；1D 卡尔曼更新，稳健且不被斜率自由度污染。
func (r *rlsState) observeOffset(x, y, measVar, lambda, qB float64) {
	r.P[0] += qB * qB
	denom := r.P[0] + measVar
	if denom <= 0 {
		return
	}
	k := r.P[0] / denom
	r.theta[0] += k * ((y - x) - r.theta[0])
	r.theta[1] = 1
	r.P[0] = (1 - k) * r.P[0] / lambda
	if r.P[0] < 1e-12 {
		r.P[0] = 1e-12
	}
}

// ClockAligner 各路时钟模型集合 + 校准暂存队列（调用方须在引擎锁内使用）。
type ClockAligner struct {
	cfg    alignParams
	models map[SensorKind]*clockModel
	rls    map[SensorKind]*rlsState
	held   map[SensorKind][]Observation
}

// alignParams 为时钟对齐所需的归一化参数（由 Config 派生）。
type alignParams struct {
	enabled            bool
	requireCalibration bool
	minSamples         int
	confirmSamples     int
	activateAt         float64 // 秒
	maxOffset          float64 // 秒
	maxSkew            float64
	forgetting         float64
	initBStd           float64
	initSStd           float64
	qB                 float64
	qS                 float64
	maxHeld            int
	learnGate          float64 // 正常学习允许的最大 |τ|（秒）
	bootstrapGate      float64 // 冷启动学习允许的最大 |τ|（秒，更宽）
	measStd            float64 // 时钟样本测量噪声标定（秒）
	measVarFloor       float64 // τ 测量方差下界（秒²），防止虚假高权重
	robustSigma        float64 // 归一化离群剔除系数
	minLinVel          float64
	minAngVel          float64
	skewActivate       float64 // 差分频偏显著阈值（无量纲）
	candAlpha          float64
	activateSNR        float64 // 激活所需信噪比（|均值|/std）
}

// NewClockAligner 创建对齐器。
func NewClockAligner(cfg Config) *ClockAligner {
	p := alignParamsFromConfig(cfg)
	a := &ClockAligner{
		cfg:    p,
		models: make(map[SensorKind]*clockModel),
		rls:    make(map[SensorKind]*rlsState),
		held:   make(map[SensorKind][]Observation),
	}
	return a
}

// ensure 建立某路模型；firstSeenInited 标识引擎是否已初始化
// （初始化前出现的路给 provisional 恒等先验，初始化后才出现的路
// 必须先校准，冷启动期观测不可信）。
func (a *ClockAligner) ensure(k SensorKind, stamp time.Time, firstSeenInited bool) *clockModel {
	m, ok := a.models[k]
	if ok {
		return m
	}
	m = &clockModel{kind: k, anchor: stamp}
	if !a.cfg.enabled || k == SensorIMU {
		m.phase = phaseReference
	} else if k == SensorAbsolutePose && a.cfg.requireCalibration {
		// 严格模式：位姿路上电即先校准（位置内容可独立辨识时钟）。
		m.phase = phaseCalibrating
	} else if firstSeenInited {
		// 任何路在引擎初始化后才首次出现：没有先验依据，先校准。
		m.phase = phaseCalibrating
	} else {
		// 默认：恒等先验，观测立即以“无偏移”参与，同时在线辨识；
		// 偏移证据连续显著后无缝切换到估计模型。
		m.phase = phaseProvisional
	}
	a.models[k] = m
	rl := newRLS(a.cfg.initBStd, a.cfg.initSStd)
	a.rls[k] = &rl
	return m
}

// mapTo 将传感器时间戳换算为统一时间（active 用估计模型，其余恒等）。
// 返回换算时间、该路是否可信；越界粘滞状态在此也会被复核。
func (a *ClockAligner) mapTo(m *clockModel, stamp time.Time) (time.Time, bool) {
	switch m.phase {
	case phaseReference, phaseProvisional:
		// 恒等先验：未激活前一律精确恒等，排序/推进/陈旧与旧版本逐位一致。
		return stamp, true
	case phaseCalibrating:
		return stamp, false
	case phaseOutOfRange:
		return stamp, false
	case phaseActive:
		r := a.rls[m.kind]
		if r.theta[1] <= 1e-3 {
			a.markOutOfRange(m, "active skew collapsed")
			return stamp, false
		}
		src := stamp.Sub(m.anchor).Seconds()
		return stamp.Add(time.Duration((r.theta[0] + (r.theta[1]-1)*src) * float64(time.Second))), true
	}
	return stamp, false
}

// Align 按传感器类型换算；未登记的路视为恒等可信（防御性）。
func (a *ClockAligner) Align(k SensorKind, stamp time.Time) (time.Time, bool) {
	m, ok := a.models[k]
	if !ok {
		return stamp, true
	}
	return a.mapTo(m, stamp)
}

// keyOf 供重排缓冲使用的有效时间键。
func (a *ClockAligner) keyOf(o Observation) time.Time {
	t, _ := a.Align(o.Kind, o.Stamp)
	return t
}

// Trusted 报告某路换算当前是否可信。
func (a *ClockAligner) Trusted(k SensorKind) bool {
	m, ok := a.models[k]
	if !ok {
		return true
	}
	return m.phase == phaseReference || m.phase == phaseProvisional || m.phase == phaseActive
}

// Phase 查询某路阶段（未登记按 reference 处理）。
func (a *ClockAligner) Phase(k SensorKind) clockPhase {
	if m, ok := a.models[k]; ok {
		return m.phase
	}
	return phaseReference
}

// currentOffset 返回某路在最新读数处的当前偏移估计（秒，读数−物理）。
func (a *ClockAligner) currentOffset(k SensorKind) float64 {
	m, ok := a.models[k]
	if !ok {
		return 0
	}
	r := a.rls[k]
	return r.theta[0] + (r.theta[1]-1)*m.maxSrc
}

// CurrentOffset 导出 currentOffset（秒）。
func (a *ClockAligner) CurrentOffset(k SensorKind) float64 { return a.currentOffset(k) }

// inRange 复核偏移范围与频偏范围（覆盖已观测的整个源时间区间）。
func (a *ClockAligner) inRange(m *clockModel) (bool, string) {
	r := a.rls[m.kind]
	b, s := r.theta[0], r.theta[1]
	off0 := b
	offN := b + (s-1)*m.maxSrc
	if math.Abs(off0) > a.cfg.maxOffset || math.Abs(offN) > a.cfg.maxOffset {
		return false, "align offset beyond trusted range"
	}
	if math.Abs(s-1) > a.cfg.maxSkew {
		return false, "clock skew beyond trusted range"
	}
	return true, ""
}

func (a *ClockAligner) markOutOfRange(m *clockModel, detail string) {
	m.phase = phaseOutOfRange
}

// Observe 追加一个时钟样本并推进 RLS 与阶段机。返回阶段是否发生变化。
func (a *ClockAligner) Observe(k SensorKind, smp clockSample) clockPhase {
	m, ok := a.models[k]
	if !ok || m.phase == phaseReference || m.phase == phaseOutOfRange {
		if !ok {
			return phaseReference
		}
		return m.phase
	}
	if math.IsNaN(smp.src) || math.IsNaN(smp.uni) || smp.measVar <= 0 {
		return m.phase
	}
	dt := smp.src - m.maxSrc
	if dt < 0 {
		dt = 0
	}
	r := a.rls[k]
	if m.phase == phaseActive {
		// 激活后：二维 RLS 跟踪偏移与斜率。
		r.observe(smp.src, smp.uni, smp.measVar, dt, a.cfg.forgetting, a.cfg.qB, a.cfg.qS)
	} else {
		// 恒等先验/校准期：仅累计稳健偏移证据，不改动 theta（保持精确
		// 恒等，零漂移输入逐位不变）；确认瞬间再一次性吸收。
		cand := smp.uni - smp.src // 本样本读数−物理（偏移）
		if m.n == 0 {
			m.offCand = cand
		} else {
			m.offCand = (1-a.cfg.candAlpha)*m.offCand + a.cfg.candAlpha*cand
		}
		// Welford 累计均值/方差，用于显著性的统计检验。
		m.candCnt++
		delta := cand - m.candMean
		m.candMean += delta / float64(m.candCnt)
		m.candM2 += delta * (cand - m.candMean)
	}
	if smp.src > m.maxSrc {
		m.maxSrc = smp.src
	}
	m.n++

	// 显著性：候选偏移均值超过阈值，且相对样本标准差有足够信噪比。
	// 这区分真实时钟偏移（持续同号、SNR 高）与模型/噪声误差
	// （均值小或被大标准差淹没），零漂移因此永不被激活。
	significant := false
	if m.candCnt >= a.cfg.minSamples {
		mean := m.candMean
		std := math.Sqrt(m.candM2 / float64(m.candCnt-1))
		snrGate := std*a.cfg.activateSNR + 1e-6
		significant = math.Abs(mean) >= a.cfg.activateAt &&
			math.Abs(mean) >= snrGate
		m.lastSign = math.Copysign(1, mean)
	}
	if significant {
		m.confirm++
	} else {
		m.confirm = 0
	}

	// 确认瞬间把统计均值作为偏移吸收进 theta（一次性），符号映射到
	// “读数−物理”约定（cand=uni−src 为负）。
	tryActivate := func() {
		if m.confirm < a.cfg.confirmSamples {
			return
		}
		r.theta[0] = -m.candMean
		if ok, _ := a.inRange(m); !ok {
			a.markOutOfRange(m, "")
			return
		}
		m.phase = phaseActive
	}
	switch m.phase {
	case phaseCalibrating:
		tryActivate()
		if m.phase == phaseCalibrating && m.n >= a.cfg.minSamples {
			m.phase = phaseProvisional // 样本够但不显著：恒等先验
		}
	case phaseProvisional:
		tryActivate()
	case phaseActive:
		if ok, _ := a.inRange(m); !ok {
			a.markOutOfRange(m, "")
		}
	}
	return m.phase
}

// ObserveDelta 用一条“(读数时刻 src, 推断物理时刻 uni)”观测更新差分频偏。
// 内部只取相邻观测的间隔比：s = Δsrc/Δuni。冷启动期据此稳健判斜率，
// 绝对截距 b 仍由 Observe 的 RLS 精修。返回更新后的阶段。
func (a *ClockAligner) ObserveDelta(k SensorKind, src, uni, measVar float64) clockPhase {
	m, ok := a.models[k]
	if !ok || m.phase == phaseReference || m.phase == phaseOutOfRange {
		if !ok {
			return phaseReference
		}
		return m.phase
	}
	// 入参 (src, uni) 即本观测的“读数时间、物理时间”（相对该路锚点，秒）。
	// 与上一配对点之差给出读数/物理间隔，比值即时钟速率。
	if m.hasPair {
		dSrc := src - m.lastSrc
		dUni := uni - m.lastUni
		if dSrc > 1e-3 && dUni > 1e-3 {
			ratio := dSrc / dUni
			if ratio > 0.5 && ratio < 2.0 { // 鲁棒区间
				if m.skewN == 0 {
					m.skewEMA = ratio
				} else {
					const alpha = 0.2
					m.skewEMA = (1-alpha)*m.skewEMA + alpha*ratio
				}
				m.skewN++
				m.lastRatios = append(m.lastRatios, ratio)
			}
		}
	}
	m.lastSrc, m.lastUni, m.hasPair = src, uni, true
	// provisional/active 阶段：把差分斜率作为对 s 的直接观测喂入 RLS。
	// 构造在该路当前最大读数处的等效点 (x, y)，使 y = b + s·x 恰反映
	// 新斜率而不改变当前偏移估计：y = theta0 + ratio·x。
	if m.phase == phaseProvisional || m.phase == phaseActive {
		x := math.Max(m.maxSrc, src)
		if x > 0 {
			r := a.rls[k]
			y := r.theta[0] + m.skewEMA*x
			r.observe(x, y, measVar*100, 0, a.cfg.forgetting, a.cfg.qB, a.cfg.qS)
			if src > m.maxSrc {
				m.maxSrc = src
			}
			m.n++
			a.refreshPhase(m)
			return m.phase
		}
	}
	// 冷启动校准：频偏证据充分后可仅靠斜率激活（b 暂置 0，由 RLS 精修）。
	if m.phase == phaseCalibrating && m.skewN >= a.cfg.minSamples {
		if math.Abs(m.skewEMA-1) > a.cfg.skewActivate {
			r := a.rls[k]
			r.theta[1] = m.skewEMA
			if ok2, _ := a.inRange(m); !ok2 {
				a.markOutOfRange(m, "")
				return m.phase
			}
			m.phase = phaseActive
			m.confirm = a.cfg.confirmSamples
		}
	}
	return m.phase
}

// refreshPhase 依据当前 RLS 估计更新阶段机（偏移显著性 + 范围）。
func (a *ClockAligner) refreshPhase(m *clockModel) {
	r := a.rls[m.kind]
	offNow := r.theta[0] + (r.theta[1]-1)*m.maxSrc
	significant := math.Abs(offNow) >= a.cfg.activateAt
	if significant {
		m.confirm++
	} else if m.confirm > 0 {
		m.confirm--
	}
	switch m.phase {
	case phaseProvisional:
		if m.n >= a.cfg.minSamples && m.confirm >= a.cfg.confirmSamples {
			if ok, _ := a.inRange(m); !ok {
				a.markOutOfRange(m, "")
				return
			}
			m.phase = phaseActive
		}
	case phaseActive:
		if ok, _ := a.inRange(m); !ok {
			a.markOutOfRange(m, "")
		}
	}
}

// HoldForCalibration 暂存校准期观测；队列满返回 false（由上层拒绝）。
func (a *ClockAligner) HoldForCalibration(o Observation) bool {
	if len(a.held[o.Kind]) >= a.cfg.maxHeld {
		return false
	}
	a.held[o.Kind] = append(a.held[o.Kind], o)
	return true
}

// Held 返回某路暂存观测（按原始时间戳排序的副本）。
func (a *ClockAligner) Held(k SensorKind) []Observation {
	q := a.held[k]
	out := make([]Observation, len(q))
	copy(out, q)
	return out
}

// ReplaceHeld 替换某路暂存队列（引擎处理后保留“未来”观测）。
func (a *ClockAligner) ReplaceHeld(k SensorKind, remain []Observation) {
	if len(remain) == 0 {
		a.held[k] = nil
		return
	}
	a.held[k] = remain
}

// ClearHeld 清空某路暂存（模型转为可信后，旧观测不再回放）。
func (a *ClockAligner) ClearHeld(k SensorKind) { a.held[k] = nil }

// HeldLen 某路暂存数量。
func (a *ClockAligner) HeldLen(k SensorKind) int { return len(a.held[k]) }

// OutOfRangeDetail 越界原因说明。
func (a *ClockAligner) OutOfRangeDetail(k SensorKind) string {
	m := a.models[k]
	if m == nil {
		return ""
	}
	ok, why := a.inRange(m)
	if ok {
		return "clock model marked out of range"
	}
	return why
}

// Status 导出某路诊断状态。
func (a *ClockAligner) Status(k SensorKind) AlignStatus {
	m, ok := a.models[k]
	if !ok {
		return AlignStatus{Kind: k, Phase: "reference", Trusted: true, Mapped: false}
	}
	r := a.rls[k]
	st := AlignStatus{
		Kind:      k,
		Phase:     m.phase.String(),
		Samples:   m.n,
		Confirmed: m.confirm,
		Skew:      r.theta[1] - 1,
		Trusted:   m.phase == phaseReference || m.phase == phaseProvisional || m.phase == phaseActive,
		Mapped:    m.phase == phaseActive,
	}
	if m.phase == phaseActive {
		src := m.maxSrc
		st.Offset = time.Duration((r.theta[0] + (r.theta[1]-1)*src) * float64(time.Second))
	}
	if m.phase == phaseOutOfRange {
		st.Detail = a.OutOfRangeDetail(k)
	}
	return st
}
