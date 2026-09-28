package fusion

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// maxDecisionLog 判定日志上限，超出滚动丢弃最旧记录。
const maxDecisionLog = 4096

// obsProcessCost 是处理单条观测的预算成本（微秒），用于确定性的
// 处理预算判定：cost·batchSize 超过 MaxProcessTime 即整批拒绝。
// 取经验上限值而非实测墙钟，保证相同输入的判定与结果严格可复现。
const obsProcessCost = 500 * time.Microsecond

// Engine 多传感器融合引擎：并发安全，预测/更新/快照交错一致。
//
// 摄入分两阶段：
//  1. 在线时间对齐校验 + 规模/积压/预算三类边界校验，入重排缓冲
//     （重复/倒退/陈旧/漂移越界/对齐不可信/超限在此拒绝）；
//  2. 按 (对齐后统一时间, 序号) 全序出队消费，门限判定在此发生，
//     同时每路时钟模型仅由“已排序、已过门限”的观测在线辨识。
//
// 时间语义：状态时间只沿统一时间轴单调前进；时钟模型更新只影响
// 尚未消费观测的换算，绝不回放已处理时间段。无漂移输入下非参考路
// 恒为精确恒等映射，融合结果与未开启对齐时逐位一致。
// 所有状态变更在写锁内原子完成，快照读取走读锁。
type Engine struct {
	mu        sync.RWMutex
	cfg       Config
	est       *Estimator
	buf       *ReorderBuffer
	align     *ClockAligner
	inited    bool
	initObs   []Observation
	lastSrc   map[SensorKind]time.Time // 各路最近有效观测的统一时间
	degraded  map[SensorKind]bool
	decisions []Decision // 判定日志（有界）
	stats     Stats

	// 最近 IMU 输入（机体系），供里程计时钟学习的加速度速率使用。
	lastGyro Vec3
	lastAcc  Vec3
	hasIMU   bool

	// 预初始化引导：IMU 建立的临时估计器，仅用于在正式初始化前
	// 在“正确的状态时间”上学习各路时钟，不承载任何融合结果。
	// 首个校准位姿观测会把引导状态重锚到该位姿，避免长程纯 IMU 发散。
	boot      *Estimator
	bootStart time.Time
	hasBoot   bool

	// learnSignificantOnly 为真时，时钟学习只接受偏移显著的样本，
	// 用于保证零漂移输入不产生任何模型更新（逐位恒等）。
	learnSignificantOnly bool
	calAnchor            map[SensorKind]calAnchorPt
	hasBootVel           bool
	lastBootLinVel       Vec3
}

// NewEngine 创建引擎。配置中的零值上限取默认值。
func NewEngine(cfg Config) *Engine {
	if cfg.Limits.MaxPendingObs <= 0 {
		cfg.Limits.MaxPendingObs = 1024
	}
	if cfg.Limits.MaxBatchSize <= 0 {
		cfg.Limits.MaxBatchSize = 256
	}
	if cfg.Limits.MaxProcessTime <= 0 {
		cfg.Limits.MaxProcessTime = 100 * time.Millisecond
	}
	if cfg.GateChi2 <= 0 {
		cfg.GateChi2 = 16.0 // 约 3σ（6 维）
	}
	if cfg.StaleAfter <= 0 {
		cfg.StaleAfter = 500 * time.Millisecond
	}
	if cfg.DegradeAfter <= 0 {
		cfg.DegradeAfter = 2 * time.Second
	}
	if cfg.MaxClockDrift <= 0 {
		cfg.MaxClockDrift = 5 * time.Second
	}
	if cfg.DegradedNoiseInflate <= 0 {
		cfg.DegradedNoiseInflate = 100
	}
	if cfg.InitWindow <= 0 {
		cfg.InitWindow = 3
	}
	a := NewClockAligner(cfg)
	e := &Engine{
		cfg:      cfg,
		align:    a,
		lastSrc:  make(map[SensorKind]time.Time),
		degraded: make(map[SensorKind]bool),
	}
	e.buf = NewReorderBuffer(cfg.Limits.MaxPendingObs, func(o Observation) time.Time {
		k, _ := a.Align(o.Kind, o.Stamp)
		return k
	})
	e.stats.RejectedByReason = make(map[RejectReason]int64)
	return e
}

// Ingest 摄入一批观测（可乱序），返回每条观测的判定结果。
//
// 三类边界在产生任何状态变更前整批拒绝、不留部分更新，且原因可区分：
//   - 单批规模超 MaxBatchSize：limit_exceeded；
//   - 入队后积压超 MaxPendingObs：backlog_full；
//   - 处理预算 cost·n 超 MaxProcessTime：budget_exceeded。
//
// 被缓冲等待重排的观测返回 Accepted=true 且 Detail 含 "buffered"。
func (e *Engine) Ingest(obs []Observation) []Decision {
	e.mu.Lock()
	defer e.mu.Unlock()

	n := len(obs)
	// 边界 1：单批规模（不触碰任何内部状态）。
	if n > e.cfg.Limits.MaxBatchSize {
		out := make([]Decision, n)
		for i, o := range obs {
			out[i] = e.record(Decision{Obs: o, Reason: ReasonBatchTooLarge,
				Detail: fmt.Sprintf("batch %d > max %d", n, e.cfg.Limits.MaxBatchSize)})
		}
		return out
	}
	// 边界 2：积压容量（本批全部入队后不得超过上限）。
	if e.buf.Len()+n > e.cfg.Limits.MaxPendingObs {
		out := make([]Decision, n)
		for i, o := range obs {
			out[i] = e.record(Decision{Obs: o, Reason: ReasonBacklogFull,
				Detail: fmt.Sprintf("pending %d + batch %d > max %d",
					e.buf.Len(), n, e.cfg.Limits.MaxPendingObs)})
		}
		return out
	}
	// 边界 3：确定性处理预算。
	budget := obsProcessCost * time.Duration(n)
	if budget > e.cfg.Limits.MaxProcessTime {
		out := make([]Decision, n)
		for i, o := range obs {
			out[i] = e.record(Decision{Obs: o, Reason: ReasonBudgetExceeded,
				Detail: fmt.Sprintf("cost %v for %d obs > budget %v",
					budget, n, e.cfg.Limits.MaxProcessTime)})
		}
		return out
	}

	out := make([]Decision, 0, n)
	for _, o := range obs {
		out = append(out, e.admit(o))
	}
	e.drain()
	return out
}

// admit 校验单条观测并入缓冲（须持写锁）。所有时间判定使用对齐后
// 的统一时间；对齐不可信时按可区分原因暂不采用。
func (e *Engine) admit(o Observation) Decision {
	m := e.align.ensure(o.Kind, o.Stamp, e.inited)
	phase := m.phase

	// 对齐不可信：冷启动样本不足 → 进有界校准队列（暂不采用、不排序、
	// 不推进）；漂移越界 → 粘滞拒绝。两种原因明确可区分，也区别于陈旧/倒退。
	if phase == phaseCalibrating {
		n := e.align.HeldLen(o.Kind)
		if !e.align.HoldForCalibration(o) {
			return e.record(Decision{Obs: o, Reason: ReasonBacklogFull,
				Detail: fmt.Sprintf("align calibration queue full (%d)", n)})
		}
		e.notePending()
		return e.record(Decision{Obs: o, Reason: ReasonAlignCalibrating,
			Detail: fmt.Sprintf("alignment not trusted yet; learning %d/%d samples",
				n+1, e.align.cfg.minSamples)})
	}
	if phase == phaseOutOfRange {
		return e.record(Decision{Obs: o, Reason: ReasonAlignOutOfRange,
			Detail: e.align.OutOfRangeDetail(o.Kind)})
	}

	key, trusted := e.align.Align(o.Kind, o.Stamp)
	if !trusted {
		return e.record(Decision{Obs: o, Reason: ReasonTimeUnavailable, Detail: "alignment unavailable"})
	}

	// 初始化后做漂移/陈旧检查；初始化前仅排序去重。
	if e.inited {
		drift := key.Sub(e.est.Nominal.Stamp)
		if drift < 0 {
			drift = -drift
		}
		if drift > e.cfg.MaxClockDrift {
			return e.record(Decision{Obs: o, Reason: ReasonTimeUnavailable,
				Detail: fmt.Sprintf("aligned clock drift %v > max %v", drift, e.cfg.MaxClockDrift)})
		}
		// 陈旧检查（统一时间）。provisional 路在两种情况下放宽到
		// MaxClockDrift 以便学习：①已学到显著偏移证据；②尚处该路
		// 引导学习窗口（前 minSamples 条）。两种放宽准入的观测在消费
		// 阶段只用于学习、不参与状态更新，直到映射可信。无偏移路严格
		// 拒绝，保持零漂移与旧版本逐位一致。
		if key.Before(e.est.Nominal.Stamp.Add(-e.cfg.StaleAfter)) {
			lag := e.est.Nominal.Stamp.Sub(key)
			hasOffset := math.Abs(e.align.currentOffset(o.Kind)) >= e.align.cfg.activateAt
			if !(phase == phaseProvisional && hasOffset &&
				lag < e.cfg.MaxClockDrift) {
				return e.record(Decision{Obs: o, Reason: ReasonStale,
					Detail: fmt.Sprintf("aligned %v older than state %v by > %v",
						key, e.est.Nominal.Stamp, e.cfg.StaleAfter)})
			}
		}
	}

	// 重复判定：(Kind,Stamp,Seq) 完全相同的观测绝不重复入缓冲。
	if e.buf.Contains(o) {
		return e.record(Decision{Obs: o, Reason: ReasonDuplicate,
			Detail: "identical observation already pending in buffer"})
	}
	if r := e.buf.Classify(o); r == ReasonDuplicate {
		return e.record(Decision{Obs: o, Reason: r,
			Detail: "rejected by reorder buffer ordering"})
	}
	// 倒退判定分两类：
	//   - 读数仅小幅落后（≤StaleAfter）：普通乱序/重复，刚性拒绝，
	//     与旧版本逐位一致；
	//   - 读数落后在 (StaleAfter, MaxClockDrift)：疑似时钟偏移特征，
	//     provisional 路放宽准入以便学习；真正落到统一时间过去的观测
	//     在消费阶段按对齐键拒绝，状态绝不回退。
	if r := e.buf.Classify(o); r == ReasonTimestampRegress {
		lag := e.est.Nominal.Stamp.Sub(key)
		// 仅当该路已学到显著偏移、且落后在可信漂移范围内时放宽，
		// 让偏移校正后的观测有机会重排；零漂移/未学到偏移的路严格
		// 拒绝（保持与旧版本逐位一致）。
		hasOffset := math.Abs(e.align.currentOffset(o.Kind)) >= e.align.cfg.activateAt
		relaxed := phase == phaseProvisional && hasOffset &&
			lag < e.cfg.MaxClockDrift
		if !relaxed {
			return e.record(Decision{Obs: o, Reason: r,
				Detail: "rejected by reorder buffer ordering"})
		}
	}
	if !e.buf.Push(o) {
		return e.record(Decision{Obs: o, Reason: ReasonBacklogFull, Detail: "reorder buffer full"})
	}
	// 读数快的位姿在“到达（读数=状态+偏移）”时，其物理内容≈当前状态，
	// 此刻的创新投影干净地给出偏移 −δ。仅当投影偏移达到显著阈值才学习
	// （零漂移时投影为零或被噪声门限剔除，保证逐位恒等）。纯时钟学习，
	// 不预测、不改变融合状态。
	if e.inited && o.Kind == SensorAbsolutePose &&
		e.align.Phase(o.Kind) == phaseProvisional &&
		key.After(e.est.Nominal.Stamp) &&
		key.Sub(e.est.Nominal.Stamp) <= e.cfg.MaxClockDrift {
		e.prelearnIfSignificant(o)
	}
	e.stats.Ingested++
	e.notePending()
	return Decision{Obs: o, Accepted: true, Reason: ReasonAccepted, Detail: "buffered"}
}

// drain 按统一时间全序消费缓冲中到期观测，并穿插处理校准队列。
func (e *Engine) drain() {
	for {
		// 校准队列：用不晚于当前状态时间的观测做无状态学习。
		e.processHeld()

		o, ok := e.buf.Peek()
		if !ok {
			return
		}
		if e.cfg.ReorderDelay > 0 {
			maxKey := e.buf.MaxStamp()
			k, _ := e.align.Align(o.Kind, o.Stamp)
			if k.After(maxKey.Add(-e.cfg.ReorderDelay)) {
				return
			}
		}
		e.buf.Pop()
		e.consume(o)
	}
}

// processHeld 为各路校准队列中“已到期”的观测生成时钟样本。
// 不预测、不更新融合状态：在当前（引导或正式）状态时间评估创新并学习。
func (e *Engine) processHeld() {
	st := e.learningState()
	if st == nil {
		return // 还没有任何 IMU，无法在时间轴上评估
	}
	for k := phaseCalibratingKindStart; k <= SensorAbsolutePose; k++ {
		if e.align.Phase(k) != phaseCalibrating {
			continue
		}
		q := e.align.Held(k)
		var remain []Observation
		gate := time.Duration(e.align.cfg.bootstrapGate * float64(time.Second))
		for _, o := range q {
			// 引导期时钟学习：引导态是“IMU 姿态 + 里程计/位姿速度”的轻量
			// 跟踪器，专门提供在正确物理时间评估的运动速率。位姿观测在读数
			// 时刻（=物理时刻+偏移）参与，残差投影给出偏移 τ。
			if o.Kind == SensorOdometry {
				e.feedBootVelocity(o)
			}
			if o.Stamp.After(st.Nominal.Stamp) {
				remain = append(remain, o) // 物理内容尚未到达
				continue
			}
			if o.Stamp.Before(st.Nominal.Stamp.Add(-gate)) {
				continue // 过旧丢弃
			}
			if o.Kind == SensorAbsolutePose {
				// 校准时钟学习用专用的“锚定 + 速度外推”引导位置：
				// 从上一已见位姿内容出发，用里程计世界系速度外推到本观测
				// 读数时刻，避免 IMU 平面长程积分漂移。构造一个临时估计态。
				if vecNorm(st.Nominal.Velocity) < 0.1 {
					remain = append(remain, o)
					continue
				}
				cst := e.calibrationEstimator(o)
				if iv := evaluateFor(cst, o, e.clockLearnGate()); iv.passed {
					e.learnFrom(cst, o, o.Stamp, false, iv)
					e.anchorCalibration(o)
				}
			}
		}
		e.align.ReplaceHeld(k, remain)
	}
}

// calibrationEstimator 构造一条位姿观测专用的轻量校准估计态：
// 位置从上一已见位姿内容按当前世界系速度线性外推到本观测读数时刻，
// 姿态用 IMU 引导姿态。它不做长程 IMU 平面积分，只服务时钟偏移辨识。
// 每次学习后由 anchorCalibration 把该观测内容记为新锚点。
func (e *Engine) calibrationEstimator(o Observation) *Estimator {
	st := e.learningState()
	// 世界系速度优先取最近里程计（瞬时速度读数不依赖时间戳，最准确），
	// 引导姿态把机体系速度转到世界系；无里程计再退回引导态速度。
	vel := st.Nominal.Velocity
	if e.hasBootVel {
		vel = QuatRotate(st.Nominal.Rotation, e.lastBootLinVel)
	}
	pos, rot := Vec3{}, st.Nominal.Rotation
	if a, ok := e.calAnchor[o.Kind]; ok {
		dt := o.Stamp.Sub(a.t).Seconds()
		pos = vecAdd(a.p, vecScale(vel, dt))
		rot = a.q
	}
	var cov [9]float64
	for i := range cov {
		cov[i] = 1.0
	}
	return NewEstimator(State{Position: pos, Velocity: vel,
		Rotation: rot, Stamp: o.Stamp}, cov)
}

// anchorCalibration 用一条已用于学习的位姿内容更新该路校准锚点。
func (e *Engine) anchorCalibration(o Observation) {
	if e.calAnchor == nil {
		e.calAnchor = make(map[SensorKind]calAnchorPt)
	}
	e.calAnchor[o.Kind] = calAnchorPt{t: o.Stamp, p: o.Pose.Position, q: QuatNorm(o.Pose.Rotation)}
}

type calAnchorPt struct {
	t time.Time
	p Vec3
	q Quat
}

// learningState 返回当前用于时钟学习的估计器：初始化前用引导估计器。
func (e *Engine) learningState() *Estimator {
	if e.inited {
		return e.est
	}
	return e.boot
}

// phaseCalibratingKindStart 校准队列扫描起点（IMU 恒为参考，跳过）。
const phaseCalibratingKindStart = SensorOdometry

// consume 消费一条已排序观测（须持写锁）。进入此处的非 IMU 路
// 必定已经通过对齐可信度检查（calibrating 在 admit 阶段被分流）。
func (e *Engine) consume(o Observation) {
	if !e.inited {
		// 仅 IMU 与已可信路会进入这里。用 IMU 维护引导时间轴，
		// 让 processHeld 能在正确的状态时间学习其他路时钟。
		if o.Kind == SensorIMU {
			e.advanceBoot(o)
		}
		e.initObs = append(e.initObs, o)
		hasPose := false
		for _, io := range e.initObs {
			if io.Kind == SensorAbsolutePose {
				hasPose = true
				break
			}
		}
		// 初始化要求收满窗口且含绝对位姿；无位姿时按观测数兜底，
		// 若配置了 InitFallbackAfter（>0）还需满足最短等待时长，
		// 以便各路时钟校准攒样本。
		fallbackCount := len(e.initObs) >= 3*e.cfg.InitWindow
		if fallbackCount && e.cfg.InitFallbackAfter > 0 && e.hasBoot && o.Kind == SensorIMU {
			fallbackCount = o.Stamp.Sub(e.bootStart) >= e.cfg.InitFallbackAfter
		}
		ready := (len(e.initObs) >= e.cfg.InitWindow && hasPose) || fallbackCount
		if !ready {
			e.record(Decision{Obs: o, Reason: ReasonInsufficient,
				Detail: fmt.Sprintf("init window %d/%d (pose=%v)",
					len(e.initObs), e.cfg.InitWindow, hasPose)})
			return
		}
		e.initialize()
		e.record(Decision{Obs: o, Accepted: true, Reason: ReasonAccepted, Detail: "initialized"})
		// 初始化后立即用当前触发观测做一次时钟学习（不更新状态），
		// 让有初始偏移的路从第一条观测就开始辨识。
		if o.Kind == SensorAbsolutePose && e.align.Phase(o.Kind) == phaseProvisional {
			e.prelearnAtCurrent(o)
		}
		return
	}

	key, trusted := e.align.Align(o.Kind, o.Stamp)
	if !trusted {
		e.record(Decision{Obs: o, Reason: ReasonAlignOutOfRange,
			Detail: e.align.OutOfRangeDetail(o.Kind)})
		return
	}
	// 恒等先验位姿路冷启动期：读数换算键不晚于状态（典型于“读数快”），
	// 在当前状态时间只做时钟学习、不推进也不更新状态（防止回退/重复消费）。
	if o.Kind == SensorAbsolutePose && e.align.Phase(o.Kind) == phaseProvisional &&
		!key.After(e.est.Nominal.Stamp) {
		e.prelearnAtCurrent(o)
		return
	}
	// 已激活/参考路：换算键不得落到状态过去（保证时间语义单调）。
	if !key.After(e.est.Nominal.Stamp) {
		if key.Before(e.est.Nominal.Stamp.Add(-e.cfg.StaleAfter)) {
			e.record(Decision{Obs: o, Reason: ReasonStale,
				Detail: "aligned key fell behind state after model update"})
			return
		}
		e.record(Decision{Obs: o, Reason: ReasonTimestampRegress,
			Detail: "aligned key not after state time after model update"})
		return
	}

	dt := key.Sub(e.est.Nominal.Stamp).Seconds()
	angWorld := QuatRotate(e.est.Nominal.Rotation, e.lastGyro)
	if o.Kind == SensorIMU {
		e.est.Predict(o, dt)
		e.hasIMU = true
		e.lastGyro = o.Gyro
		e.lastAcc = o.Accel
	} else {
		e.est.PredictConstantVelocity(dt, angWorld)
	}
	e.est.AddProcessNoise(e.cfg.ProcessNoise, dt)

	// 预测到读数时间后，对恒等先验位姿路做时钟学习：此刻状态在该读数
	// 时刻，位姿内容相对状态的位置残差 = −v·偏移，投影 τ 带符号给出
	// （读数快 τ<0）。仅更新时钟模型；随后的状态更新正常进行。
	if o.Kind == SensorAbsolutePose && e.align.Phase(o.Kind) == phaseProvisional {
		e.prelearnAtCurrent(o)
	}

	// 传感器降级检查（统一时间）。
	e.refreshDegraded(key)
	inflated := false
	if e.degraded[o.Kind] {
		inflated = true
		for i := range o.CovDiag {
			o.CovDiag[i] *= e.cfg.DegradedNoiseInflate
		}
	}

	if o.Kind == SensorIMU {
		e.lastSrc[o.Kind] = key
		e.stats.Consumed++
		e.record(Decision{Obs: o, Accepted: true, Reason: ReasonAccepted})
		return
	}

	// 时钟学习与状态更新共用同一次创新评估：先学（通过门限），
	// 再提交误差状态。门限拒绝的观测既不更新也不学习。
	d2, accepted := e.learnAndUpdate(o, key)
	if !accepted {
		e.record(Decision{Obs: o, Reason: ReasonOutlierGate, Maha: d2,
			Detail: fmt.Sprintf("mahalanobis^2 %.3f > gate %.3f", d2, e.cfg.GateChi2)})
		return
	}
	e.stats.Consumed++
	e.lastSrc[o.Kind] = key
	if inflated {
		e.record(Decision{Obs: o, Accepted: true, Reason: ReasonSensorDegraded, Maha: d2,
			Detail: fmt.Sprintf("sensor %s degraded; noise inflated x%.0f",
				o.Kind, e.cfg.DegradedNoiseInflate)})
		return
	}
	e.record(Decision{Obs: o, Accepted: true, Reason: ReasonAccepted, Maha: d2})
}

// prelearnIfSignificant 仅在投影偏移达到显著阈值且通过 SNR 门限时学习，
// 保证零漂移（投影≈0）不产生任何时钟模型更新，从而与旧版本逐位一致。
func (e *Engine) prelearnIfSignificant(o Observation) {
	saved := e.learnSignificantOnly
	e.learnSignificantOnly = true
	e.prelearnAtCurrent(o)
	e.learnSignificantOnly = saved
}

// prelearnAtCurrent 在不预测、不修改融合状态的前提下，于当前状态时间
// 评估一条恒等先验路观测的创新，提取时钟时序证据并更新该路模型。
func (e *Engine) prelearnAtCurrent(o Observation) {
	iv := evaluateFor(e.est, o, e.clockLearnGate())
	if !iv.passed {
		return
	}
	// 直接构造时钟样本：读数时间相对锚点为 src；推断的物理时刻为
	// 当前状态时间 + τ（τ 由创新对时间的一阶投影带符号给出）。
	n := e.est.Nominal
	var ev timingEvidence
	switch o.Kind {
	case SensorAbsolutePose:
		angWorld := QuatRotate(n.Rotation, e.lastGyro)
		ev = poseTimingEvidence(iv, n.Velocity, angWorld,
			e.align.cfg.minLinVel, e.align.cfg.minAngVel)

	case SensorOdometry:
		var accW Vec3
		if e.hasIMU {
			accW = vecAdd(QuatRotate(n.Rotation, e.lastAcc), Vec3{Z: -9.81})
		}
		ev = odomTimingEvidence(iv, n.Rotation, accW, e.align.cfg.minLinVel)
	default:
		return
	}
	if !ev.valid {
		return
	}
	tau := ev.tau
	if tau > e.align.cfg.learnGate || tau < -e.align.cfg.learnGate {
		return
	}
	sig2 := ev.measVar
	if v := e.align.cfg.measVarFloor; v > sig2 {
		sig2 = v
	}
	if math.Abs(tau) > e.align.cfg.robustSigma*math.Sqrt(sig2) {
		return
	}
	if e.learnSignificantOnly && math.Abs(tau) < e.align.cfg.activateAt {
		return
	}
	m := e.align.models[o.Kind]
	// 在“预测前”评估：投影 τ = T_phys − T_state（观测物理时刻相对
	// 当前状态时间）。物理时间 = T_state + τ；读数 src 与之差即偏移。
	phys := n.Stamp.Add(time.Duration(tau * float64(time.Second)))
	src := o.Stamp.Sub(m.anchor).Seconds()
	uni := phys.Sub(m.anchor).Seconds()
	// 位置创新 τ（圆周等一般运动中位置信噪比高）直接给出读数−物理偏移，
	// RLS 同时辨识截距与斜率，跟踪时变漂移。
	e.align.Observe(o.Kind, clockSample{src: src, uni: uni, measVar: sig2})
}

// learnAndUpdate 按状态门限评估创新并提交状态更新。
// 时钟学习已在消费前于正确的状态时间完成（prelearnAtCurrent 或
// processHeld），此处不再在错位时间重复学习。
func (e *Engine) learnAndUpdate(o Observation, key time.Time) (float64, bool) {
	iv := evaluateFor(e.est, o, e.cfg.GateChi2)
	if !iv.passed {
		return iv.maha2, false
	}
	e.est.commit(iv)
	return iv.maha2, true
}

// evaluateFor 按观测类型在给定估计器上做创新评估。
func evaluateFor(st *Estimator, o Observation, gate float64) innovation {
	switch o.Kind {
	case SensorOdometry:
		return st.EvaluateOdometry(o, gate)
	case SensorAbsolutePose:
		return st.EvaluateAbsolutePose(o, gate)
	default:
		return innovation{passed: false, maha2: math.Inf(1)}
	}
}

// clockLearnGate 时钟学习专用宽门限（远大于状态门限）。
func (e *Engine) clockLearnGate() float64 {
	if e.cfg.GateChi2 > 0 {
		return e.cfg.GateChi2 * 1e4
	}
	return 1e6
}

// learnFrom 从一条观测的创新（已按时钟宽门限通过）中提取时钟样本。
// consume=true 为正常消费路径；false 为冷启动校准队列的无状态学习。
func (e *Engine) learnFrom(st *Estimator, o Observation, key time.Time, consume bool, iv innovation) {
	phase := e.align.Phase(o.Kind)
	if phase == phaseReference || phase == phaseOutOfRange {
		return
	}
	n := st.Nominal
	var ev timingEvidence
	switch o.Kind {
	case SensorAbsolutePose:
		// 世界系角速率由机体系陀螺经当前姿态旋转得到。
		angWorld := QuatRotate(n.Rotation, e.lastGyro)
		ev = poseTimingEvidence(iv, n.Velocity, angWorld,
			e.align.cfg.minLinVel, e.align.cfg.minAngVel)
	case SensorOdometry:
		var accW Vec3
		if e.hasIMU {
			accW = vecAdd(QuatRotate(n.Rotation, e.lastAcc), Vec3{Z: -9.81})
		}
		ev = odomTimingEvidence(iv, n.Rotation, accW, e.align.cfg.minLinVel)
	default:
		return
	}
	if !ev.valid {
		return
	}

	// τ* = t_phys − T：对 h(t_phys)−h(T) ≈ ḣ·(t_phys−T)，标量投影
	// τ* = (rᵀW ḣ)/(ḣᵀW ḣ) 直接带符号给出（读数慢→τ>0，读数快→τ<0）。
	tau := ev.tau
	gate := e.align.cfg.learnGate
	if !consume {
		gate = e.align.cfg.bootstrapGate
	}
	if tau > gate || tau < -gate {
		return
	}
	// 测量方差融合：创新投影方差与先验标定噪声取大，防止极小观测 R
	// 给出虚假的高权重；再做归一化离群剔除（|τ| 不得超过 k·σ_total）。
	sig2 := ev.measVar
	if v := e.align.cfg.measVarFloor; v > sig2 {
		sig2 = v
	}
	sig := math.Sqrt(sig2)
	if math.Abs(tau) > e.align.cfg.robustSigma*sig {
		return
	}
	m := e.align.models[o.Kind]
	// 推断的物理（统一）时刻 = 状态时刻 + τ；再换算到该路锚点坐标。
	phys := st.Nominal.Stamp.Add(time.Duration(tau * float64(time.Second)))
	smp := clockSample{
		src:     o.Stamp.Sub(m.anchor).Seconds(),
		uni:     phys.Sub(m.anchor).Seconds(),
		measVar: sig2,
	}
	prev := phase
	newPhase := e.align.Observe(o.Kind, smp)
	if prev != phaseActive && newPhase == phaseActive {
		// 模型首次生效：清掉校准期暂存（其物理时间多已过去，
		// 回放只会被陈旧判定拒绝，没有信息价值）。
		e.align.ClearHeld(o.Kind)
	}
}

// feedBootVelocity 用一条里程计观测的机体系速度设置引导状态世界系速度。
// 瞬时速度读数不依赖时间戳，故即使该路时钟在校准中仍可用于引导运动。
func (e *Engine) feedBootVelocity(o Observation) {
	e.hasBootVel = true
	e.lastBootLinVel = o.LinVel // 机体系瞬时速度（不依赖时间戳）
	if e.boot == nil {
		return
	}
	e.boot.Nominal.Velocity = QuatRotate(e.boot.Nominal.Rotation, o.LinVel)
}

// advanceBoot 用 IMU 观测建立/推进预初始化引导估计器（不承载融合结果）。
func (e *Engine) advanceBoot(o Observation) {
	if !e.hasBoot {
		var cov [9]float64
		for i := range cov {
			cov[i] = 1.0
		}
		e.boot = NewEstimator(State{Rotation: QuatIdentity(), Stamp: o.Stamp}, cov)
		e.hasBoot = true
		e.bootStart = o.Stamp
		e.hasIMU = true
		e.lastGyro, e.lastAcc = o.Gyro, o.Accel
		return
	}
	dt := o.Stamp.Sub(e.boot.Nominal.Stamp).Seconds()
	if dt > 0 {
		e.boot.Predict(o, dt)
		e.boot.AddProcessNoise(e.cfg.ProcessNoise, dt)
	}
	e.hasIMU = true
	e.lastGyro, e.lastAcc = o.Gyro, o.Accel
}

// initialize 用缓冲的初始观测建立初始状态。
func (e *Engine) initialize() {
	var init State
	var cov [9]float64
	for i := range cov {
		cov[i] = 1.0
	}
	for _, o := range e.initObs {
		key, _ := e.align.Align(o.Kind, o.Stamp)
		if o.Kind == SensorAbsolutePose {
			init.Position = o.Pose.Position
			init.Rotation = o.Pose.Rotation
		}
		if key.After(init.Stamp) || init.Stamp.IsZero() {
			init.Stamp = key
		}
	}
	e.est = NewEstimator(init, cov)
	e.inited = true
	for _, o := range e.initObs {
		key, _ := e.align.Align(o.Kind, o.Stamp)
		e.lastSrc[o.Kind] = key
	}
	e.initObs = nil
}

// refreshDegraded 依据各路最近观测的统一时间刷新降级标记（须持写锁）。
func (e *Engine) refreshDegraded(now time.Time) {
	for k := SensorIMU; k <= SensorAbsolutePose; k++ {
		last, seen := e.lastSrc[k]
		if !seen {
			continue
		}
		e.degraded[k] = now.Sub(last) > e.cfg.DegradeAfter
	}
}

// AdvanceTo 将状态预测推进到统一时刻 t：先消费缓冲中不晚于 t 的观测，
// 再纯预测剩余区间。目标时刻不晚于当前状态时返回可区分错误。
func (e *Engine) AdvanceTo(t time.Time) Decision {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.inited && e.buf.Len() == 0 {
		return e.record(Decision{Reason: ReasonInsufficient, Detail: "not initialized"})
	}
	for {
		e.processHeld()
		o, ok := e.buf.Peek()
		if !ok {
			break
		}
		k, _ := e.align.Align(o.Kind, o.Stamp)
		if k.After(t) {
			break
		}
		e.buf.Pop()
		e.consume(o)
	}
	if !e.inited {
		return e.record(Decision{Reason: ReasonInsufficient, Detail: "not initialized"})
	}
	if !t.After(e.est.Nominal.Stamp) {
		return e.record(Decision{Reason: ReasonTimestampRegress, Detail: "advance target not after state time"})
	}
	dt := t.Sub(e.est.Nominal.Stamp).Seconds()
	e.est.PredictConstantVelocity(dt)
	e.est.AddProcessNoise(e.cfg.ProcessNoise, dt)
	e.refreshDegraded(t)
	return e.record(Decision{Accepted: true, Reason: ReasonAccepted, Detail: "advanced"})
}

// Snapshot 并发安全的状态快照。第三个返回值标识是否已初始化。
func (e *Engine) Snapshot() (State, Mat, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.inited {
		return State{}, Mat{}, false
	}
	s, c := e.est.Snapshot()
	return s, c, true
}

// Degraded 返回各路传感器当前是否处于降级状态。
func (e *Engine) Degraded() map[SensorKind]bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make(map[SensorKind]bool, len(e.degraded))
	for k, v := range e.degraded {
		out[k] = v
	}
	return out
}

// AlignStatuses 返回各路在线时间对齐状态诊断。
func (e *Engine) AlignStatuses() map[SensorKind]AlignStatus {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make(map[SensorKind]AlignStatus, 3)
	for k := SensorIMU; k <= SensorAbsolutePose; k++ {
		out[k] = e.align.Status(k)
	}
	return out
}

// Stats 返回运行统计（待处理峰值、摄入/消费量、拒绝原因分布）。
func (e *Engine) Stats() Stats {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := Stats{
		PeakPending:      e.stats.PeakPending,
		Ingested:         e.stats.Ingested,
		Consumed:         e.stats.Consumed,
		RejectedByReason: make(map[RejectReason]int64, len(e.stats.RejectedByReason)),
	}
	for k, v := range e.stats.RejectedByReason {
		out.RejectedByReason[k] = v
	}
	return out
}

// Decisions 返回判定日志副本。
func (e *Engine) Decisions() []Decision {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]Decision, len(e.decisions))
	copy(out, e.decisions)
	return out
}

// notePending 更新待处理峰值与摄入计数（须持写锁）。
func (e *Engine) notePending() {
	if p := e.buf.Len(); p > e.stats.PeakPending {
		e.stats.PeakPending = p
	}
}

// record 记录判定（有界滚动），累计拒绝统计，并返回该判定。
func (e *Engine) record(d Decision) Decision {
	if d.Reason == ReasonAccepted && d.Accepted && d.Detail == "buffered" {
		e.stats.Ingested++
	}
	if len(e.decisions) >= maxDecisionLog {
		e.decisions = e.decisions[1:]
	}
	if d.Reason != ReasonAccepted {
		e.stats.RejectedByReason[d.Reason]++
	}
	e.decisions = append(e.decisions, d)
	return d
}
