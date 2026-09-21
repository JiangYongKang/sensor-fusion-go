package fusion

import (
	"fmt"
	"sync"
	"time"
)

// maxDecisionLog 判定日志上限，超出滚动丢弃最旧记录。
const maxDecisionLog = 4096

// Engine 多传感器融合引擎：并发安全，预测/更新/快照交错一致。
//
// 摄入分两阶段：
//  1. 校验并入重排缓冲（重复/倒退/陈旧/漂移/超限在此拒绝）；
//  2. 按 (时间戳, 序号) 全序出队消费，门限判定在此发生。
//
// ReorderDelay > 0 时，时间戳晚于 (最新观测-ReorderDelay) 的观测
// 保留在缓冲中等待乱序伙伴；其余在每次 Ingest/AdvanceTo 后消费。
// 所有状态变更在写锁内原子完成，快照读取走读锁。
type Engine struct {
	mu        sync.RWMutex
	cfg       Config
	est       *Estimator
	buf       *ReorderBuffer
	inited    bool
	initObs   []Observation
	lastSrc   map[SensorKind]time.Time // 各路最近有效观测时间
	degraded  map[SensorKind]bool
	decisions []Decision // 判定日志（有界）
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
	return &Engine{
		cfg:      cfg,
		buf:      NewReorderBuffer(cfg.Limits.MaxPendingObs),
		lastSrc:  make(map[SensorKind]time.Time),
		degraded: make(map[SensorKind]bool),
	}
}

// Ingest 摄入一批观测（可乱序），返回每条观测的判定结果。
// 批大小超限时整批拒绝，不残留部分更新。
// 被缓冲等待重排的观测返回 Accepted=true 且 Detail 含 "buffered"。
func (e *Engine) Ingest(obs []Observation) []Decision {
	e.mu.Lock()
	defer e.mu.Unlock()

	deadline := time.Now().Add(e.cfg.Limits.MaxProcessTime)

	// 规模上限：整批拒绝，不产生任何部分更新。
	if len(obs) > e.cfg.Limits.MaxBatchSize {
		out := make([]Decision, len(obs))
		for i, o := range obs {
			out[i] = Decision{Obs: o, Reason: ReasonLimitExceeded,
				Detail: fmt.Sprintf("batch %d > max %d", len(obs), e.cfg.Limits.MaxBatchSize)}
		}
		return out
	}

	out := make([]Decision, 0, len(obs))
	for _, o := range obs {
		// 耗时上限：剩余观测按 LimitExceeded 拒绝，已完成的判定保留返回。
		if time.Now().After(deadline) {
			out = append(out, Decision{Obs: o, Reason: ReasonLimitExceeded,
				Detail: "process time budget exceeded"})
			continue
		}
		out = append(out, e.ingestOne(o))
	}
	e.drain()
	return out
}

// ingestOne 校验单条观测并入缓冲（须持写锁）。
func (e *Engine) ingestOne(o Observation) Decision {
	// 时钟漂移检查：观测时间不得偏离引擎当前时间超过 MaxClockDrift。
	if e.inited {
		drift := o.Stamp.Sub(e.est.Nominal.Stamp)
		if drift < 0 {
			drift = -drift
		}
		if drift > e.cfg.MaxClockDrift {
			return e.log(Decision{Obs: o, Reason: ReasonTimeUnavailable,
				Detail: fmt.Sprintf("clock drift %v > max %v", drift, e.cfg.MaxClockDrift)})
		}
	}

	// 陈旧检查：远落后于已融合时间的观测不得当作最新数据。
	if e.inited && o.Stamp.Before(e.est.Nominal.Stamp.Add(-e.cfg.StaleAfter)) {
		return e.log(Decision{Obs: o, Reason: ReasonStale,
			Detail: fmt.Sprintf("obs %v older than state %v by > %v",
				o.Stamp, e.est.Nominal.Stamp, e.cfg.StaleAfter)})
	}

	// 重复/倒退判定（相对已出队序列与缓冲内待定观测）。
	if r := e.buf.Classify(o); r == ReasonDuplicate || r == ReasonTimestampRegress {
		return e.log(Decision{Obs: o, Reason: r,
			Detail: "rejected by reorder buffer ordering"})
	}
	if e.buf.Contains(o) {
		return e.log(Decision{Obs: o, Reason: ReasonDuplicate,
			Detail: "identical observation already pending in buffer"})
	}

	// 入重排缓冲；满则按策略拒绝。
	if !e.buf.Push(o) {
		return e.log(Decision{Obs: o, Reason: ReasonLimitExceeded,
			Detail: "reorder buffer full"})
	}
	return Decision{Obs: o, Accepted: true, Reason: ReasonAccepted, Detail: "buffered"}
}

// drain 按全序消费缓冲中到期的观测（须持写锁）。
// ReorderDelay 为 0 时消费全部；否则保留时间戳晚于
// (缓冲内最大时间戳 - ReorderDelay) 的观测等待乱序伙伴。
func (e *Engine) drain() {
	for {
		o, ok := e.buf.Peek()
		if !ok {
			return
		}
		if e.cfg.ReorderDelay > 0 {
			maxStamp := e.buf.MaxStamp()
			if o.Stamp.After(maxStamp.Add(-e.cfg.ReorderDelay)) {
				return
			}
		}
		e.buf.Pop()
		e.consume(o)
	}
}

// consume 消费一条已排序观测（须持写锁）。
func (e *Engine) consume(o Observation) {
	if !e.inited {
		e.initObs = append(e.initObs, o)
		// 初始化策略：收集满 InitWindow 条且含绝对位姿时初始化；
		// 若一直无绝对位姿，累积到 3 倍窗口后以零位姿兜底初始化，
		// 保证纯 IMU/里程计场景也能启动（误差由后续观测收敛）。
		hasPose := false
		for _, io := range e.initObs {
			if io.Kind == SensorAbsolutePose {
				hasPose = true
				break
			}
		}
		ready := (len(e.initObs) >= e.cfg.InitWindow && hasPose) ||
			len(e.initObs) >= 3*e.cfg.InitWindow
		if !ready {
			e.log(Decision{Obs: o, Reason: ReasonInsufficient,
				Detail: fmt.Sprintf("init window %d/%d (pose=%v)",
					len(e.initObs), e.cfg.InitWindow, hasPose)})
			return
		}
		e.initialize()
		e.log(Decision{Obs: o, Accepted: true, Reason: ReasonAccepted,
			Detail: "initialized"})
		return
	}

	// 预测推进到观测时刻：IMU 观测驱动积分，其余观测到达前用
	// 恒定速度模型（零比力）推进。
	dt := o.Stamp.Sub(e.est.Nominal.Stamp).Seconds()
	if dt > 0 {
		if o.Kind == SensorIMU {
			e.est.Predict(o, dt)
		} else {
			e.est.PredictConstantVelocity(dt)
		}
		e.est.AddProcessNoise(e.cfg.ProcessNoise, dt)
	}

	// 传感器降级检查：缺失超时的路进入降级；降级期间的观测
	// 不直接拒绝，而是按既定倍数膨胀其观测噪声后参与更新，
	// 避免把可能陈旧的数据当作最新数据，同时允许传感器恢复。
	e.refreshDegraded(o.Stamp)
	inflated := false
	if e.degraded[o.Kind] {
		inflated = true
		for i := range o.CovDiag {
			o.CovDiag[i] *= e.cfg.DegradedNoiseInflate
		}
	}

	var d2 float64
	var ok bool
	switch o.Kind {
	case SensorIMU:
		// IMU 已在预测中消费；此处作为到达即接受的高频源。
		e.lastSrc[o.Kind] = o.Stamp
		e.log(Decision{Obs: o, Accepted: true, Reason: ReasonAccepted})
		return
	case SensorOdometry:
		d2, ok = e.est.UpdateOdometry(o, e.cfg.GateChi2)
	case SensorAbsolutePose:
		d2, ok = e.est.UpdateAbsolutePose(o, e.cfg.GateChi2)
	default:
		e.log(Decision{Obs: o, Reason: ReasonInsufficient, Detail: "unknown sensor kind"})
		return
	}
	if !ok {
		e.log(Decision{Obs: o, Reason: ReasonOutlierGate, Maha: d2,
			Detail: fmt.Sprintf("mahalanobis^2 %.3f > gate %.3f", d2, e.cfg.GateChi2)})
		return
	}
	e.lastSrc[o.Kind] = o.Stamp
	if inflated {
		e.log(Decision{Obs: o, Accepted: true, Reason: ReasonSensorDegraded, Maha: d2,
			Detail: fmt.Sprintf("sensor %s degraded; noise inflated x%.0f",
				o.Kind, e.cfg.DegradedNoiseInflate)})
		return
	}
	e.log(Decision{Obs: o, Accepted: true, Reason: ReasonAccepted, Maha: d2})
}

// initialize 用缓冲的初始观测建立初始状态。
func (e *Engine) initialize() {
	var init State
	var cov [9]float64
	for i := range cov {
		cov[i] = 1.0
	}
	// 优先使用绝对位姿观测作为初始位姿。
	for _, o := range e.initObs {
		if o.Kind == SensorAbsolutePose {
			init.Position = o.Pose.Position
			init.Rotation = o.Pose.Rotation
		}
		if o.Stamp.After(init.Stamp) || init.Stamp.IsZero() {
			init.Stamp = o.Stamp
		}
	}
	e.est = NewEstimator(init, cov)
	e.inited = true
	for _, o := range e.initObs {
		e.lastSrc[o.Kind] = o.Stamp
	}
	e.initObs = nil
}

// refreshDegraded 依据各路最近观测时间刷新降级标记（须持写锁）。
func (e *Engine) refreshDegraded(now time.Time) {
	for k := SensorIMU; k <= SensorAbsolutePose; k++ {
		last, seen := e.lastSrc[k]
		if !seen {
			continue // 从未出现的路不判降级
		}
		e.degraded[k] = now.Sub(last) > e.cfg.DegradeAfter
	}
}

// AdvanceTo 将状态预测推进到时刻 t：先消费缓冲中不晚于 t 的观测，
// 再纯预测剩余区间。目标时刻不晚于当前状态时返回可区分错误。
func (e *Engine) AdvanceTo(t time.Time) Decision {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.inited && e.buf.Len() == 0 {
		return Decision{Reason: ReasonInsufficient, Detail: "not initialized"}
	}
	// 消费所有不晚于 t 的待定观测。
	for {
		o, ok := e.buf.Peek()
		if !ok || o.Stamp.After(t) {
			break
		}
		e.buf.Pop()
		e.consume(o)
	}
	if !e.inited {
		return Decision{Reason: ReasonInsufficient, Detail: "not initialized"}
	}
	if !t.After(e.est.Nominal.Stamp) {
		return Decision{Reason: ReasonTimestampRegress, Detail: "advance target not after state time"}
	}
	dt := t.Sub(e.est.Nominal.Stamp).Seconds()
	e.est.PredictConstantVelocity(dt)
	e.est.AddProcessNoise(e.cfg.ProcessNoise, dt)
	e.refreshDegraded(t)
	return Decision{Accepted: true, Reason: ReasonAccepted, Detail: "advanced"}
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

// Decisions 返回判定日志副本。
func (e *Engine) Decisions() []Decision {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]Decision, len(e.decisions))
	copy(out, e.decisions)
	return out
}

// log 记录判定（有界滚动），并返回该判定。
func (e *Engine) log(d Decision) Decision {
	if len(e.decisions) >= maxDecisionLog {
		e.decisions = e.decisions[1:]
	}
	e.decisions = append(e.decisions, d)
	return d
}
