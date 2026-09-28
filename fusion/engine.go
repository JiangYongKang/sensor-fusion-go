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
//  1. 校验并经在线时间对齐换算后入重排缓冲（重复/倒退/陈旧/漂移/
//     对齐不可信/超限在此拒绝或暂缓）；
//  2. 按 (对齐后统一时间, 序号) 全序出队消费，门限判定在此发生。
//
// 所有顺序判定一律以对齐后的统一时间为准；未对齐可信的观测只暂缓、
// 绝不参与状态推进。ReorderDelay > 0 时额外保留乱序等待窗口。
// 所有状态变更在写锁内原子完成，快照读取走读锁。
type Engine struct {
	mu          sync.RWMutex
	cfg         Config
	est         *Estimator
	buf         *ReorderBuffer
	align       *Alignment
	inited      bool
	initObs     []Observation
	lastSrc     map[SensorKind]time.Time // 各路最近有效观测的统一时间
	degraded    map[SensorKind]bool
	decisions   []Decision // 判定日志（有界）
	pendingHigh int
	consumed    int
	rejected    int
	opsUsed     int // 当前 Ingest 调用内累计的确定性操作数
	nowFunc     func() time.Time
	sawRefKind  bool // 是否曾见到配置指定的参考路
}

// anchorProbeCost 一次时延网格搜索的折算操作成本上限（粗+精网格
// 评估点数，MaxDrift=10s 量级）。批前据此判定预算是否足以启动一批。
const anchorProbeCost = 2200

// NewEngine 创建引擎。配置中的零值上限取默认值。
func NewEngine(cfg Config) *Engine {
	if cfg.Limits.MaxPendingObs <= 0 {
		cfg.Limits.MaxPendingObs = 1024
	}
	if cfg.Limits.MaxBatchSize <= 0 {
		cfg.Limits.MaxBatchSize = 256
	}
	if cfg.Limits.MaxProcessTime <= 0 {
		cfg.Limits.MaxProcessTime = 10 * time.Second
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
	if cfg.AlignMinSamples <= 0 {
		cfg.AlignMinSamples = 12
	}
	if cfg.AlignMinSpan <= 0 {
		cfg.AlignMinSpan = 1 * time.Second
	}
	if cfg.AlignDetect <= 0 {
		cfg.AlignDetect = 200 * time.Millisecond
	}
	if cfg.AlignTolerance <= 0 {
		cfg.AlignTolerance = 25 * time.Millisecond
	}
	if cfg.AlignMaxSkew <= 0 {
		cfg.AlignMaxSkew = 0.05
	}
	if cfg.AlignWindow <= 0 {
		cfg.AlignWindow = 20 * time.Second
	}
	if cfg.AlignReleaseMargin <= 0 {
		cfg.AlignReleaseMargin = 25 * time.Millisecond
	}
	if cfg.AlignMaxDrift <= 0 {
		cfg.AlignMaxDrift = cfg.MaxClockDrift
	}
	if cfg.AlignSearchRange <= 0 {
		cfg.AlignSearchRange = 2 * time.Second
	}
	if cfg.AlignCalibrateAfter <= 0 {
		cfg.AlignCalibrateAfter = 5 * time.Second
	}
	// 参考路约定：AlignAutoRef=true（对齐开启时的默认）时自动选择
	// （IMU 优先，无 IMU 取首达路）；false 时严格等待 AlignRef 指定
	// 的路。枚举 IMU 恰为 0，因此“自动 vs 钉住 IMU”只能由布尔位区分。
	_ = AlignRefAuto
	return &Engine{
		cfg:      cfg,
		buf:      NewReorderBuffer(cfg.Limits.MaxPendingObs),
		align:    NewAlignment(cfg),
		lastSrc:  make(map[SensorKind]time.Time),
		degraded: make(map[SensorKind]bool),
		nowFunc:  time.Now,
	}
}

// Ingest 摄入一批观测（可乱序），返回每条观测的判定结果。
// 单批规模超限（limit_exceeded）、积压过多（backlog_exceeded）或
// 处理预算不足（process_budget）时整批拒绝，不残留任何部分更新；
// 拒绝后引擎仍可继续处理后续批次。
func (e *Engine) Ingest(obs []Observation) []Decision {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.opsUsed = 0

	// 规模上限：整批拒绝，不产生任何部分更新。
	if len(obs) > e.cfg.Limits.MaxBatchSize {
		out := make([]Decision, len(obs))
		for i, o := range obs {
			out[i] = e.commit(Decision{Obs: o, RawStamp: o.Stamp,
				Reason: ReasonLimitExceeded,
				Detail: fmt.Sprintf("batch %d > max %d", len(obs), e.cfg.Limits.MaxBatchSize)})
		}
		return out
	}

	// 积压上限在 ingestOne 中逐条判定：参考路（时间基准，恒等映射）无
	// 论缓冲多满都必须接收——否则缺测窗口会把参考时间轴一起饿死，造成
	// 所有路前沿永久冻结；非参考路在缓冲满时按 backlog_exceeded 拒绝。
	// 这里不再做整批预检，避免一批混合观测里的非参考路把参考路连带拒绝。

	// 确定性操作预算：批前若本批至少一条观测需要一次对齐时延搜索，
	// 而剩余预算不足以覆盖一次搜索，则整批拒绝（process_budget）。
	// MaxBatchOps<=0 时不施加操作预算（仅受宽松墙钟保护）。
	if e.cfg.Limits.MaxBatchOps > 0 && e.opsUsed+anchorProbeCost > e.cfg.Limits.MaxBatchOps {
		out := make([]Decision, len(obs))
		for i, o := range obs {
			out[i] = e.commit(Decision{Obs: o, RawStamp: o.Stamp,
				Reason: ReasonProcessBudget,
				Detail: fmt.Sprintf("op budget %d exhausted (%d used); batch of %d rejected whole",
					e.cfg.Limits.MaxBatchOps, e.opsUsed, len(obs))})
		}
		e.rejected += len(obs)
		return out
	}

	// 墙钟仅作宽松保护，正常确定性测试不受其影响。
	deadline := e.nowFunc().Add(e.cfg.Limits.MaxProcessTime)

	// 对齐层检查点：预算失败时回滚本批造成的对齐模型变动。
	cp := e.align.Checkpoint()
	var pushed []pendingItem
	out := make([]Decision, 0, len(obs))
	budgetFail := false
	for _, o := range obs {
		io := e.ingestOne(o)
		e.opsUsed += io.deltaOps
		if e.cfg.Limits.MaxBatchOps > 0 && e.opsUsed > e.cfg.Limits.MaxBatchOps {
			budgetFail = true
			break
		}
		if e.nowFunc().After(deadline) {
			budgetFail = true
			break
		}
		out = append(out, io.d)
		if io.pushed {
			pushed = append(pushed, io.p)
		}
	}
	if budgetFail {
		// 整批回滚：撤销已入缓冲项与对齐模型更新，本批统一记预算不足。
		for _, p := range pushed {
			e.buf.Remove(p.kind, p.raw, p.obs.Seq)
		}
		e.align.Restore(cp)
		e.rejected += len(obs)
		res := make([]Decision, len(obs))
		for i, o := range obs {
			res[i] = e.commit(Decision{Obs: o, RawStamp: o.Stamp,
				Reason: ReasonProcessBudget,
				Detail: fmt.Sprintf("process budget exhausted mid-batch (ops limit %d)",
					e.cfg.Limits.MaxBatchOps)})
		}
		return res
	}

	// 参考路一旦由配置指定便只等待该路出现；始终不出现时非参考观测
	// 保持 align_calibrating（明确且可区分），不做静默改路。未配置
	// 参考路（AlignRef 为未知路）时，对齐层以最先到达的一路为参考。
	// 模型可能在校准后转 locked：重判所有暂缓项并重算对齐时间。
	e.retryPending()
	e.buf.RemapAll(e.align)

	if e.buf.Len() > e.pendingHigh {
		e.pendingHigh = e.buf.Len()
	}

	// 状态推进阶段同样受预算约束；预算不足则仅保留缓冲，下批继续。
	if !e.nowFunc().After(deadline) {
		e.drain(deadline)
	}
	return out
}

// ingestOut 单条摄入结果。
type ingestOut struct {
	p        pendingItem
	d        Decision
	pushed   bool
	deltaOps int
}

// ingestOne 校验单条观测、换算统一时间并入缓冲（须持写锁）。
func (e *Engine) ingestOne(o Observation) (r ingestOut) {
	if e.cfg.EnableAlignment && o.Kind == e.cfg.AlignRef {
		e.sawRefKind = true
	}
	res := e.align.Admit(o)
	kind := o.Kind
	r.deltaOps = res.ops

	// 对齐尚不可信：明确区分“样本不足/参考未覆盖/宽限内校准”与“超可
	// 信范围”。ready=false 时观测将挂起；非参考路若缓冲已满则拒绝（保
	// 持待处理有界），参考路由对齐层保证恒等就绪、不会走到这里。
	if !res.ready {
		if e.buf.Len() >= e.cfg.Limits.MaxPendingObs {
			r.d = e.commit(Decision{Obs: o, Accepted: false, RawStamp: o.Stamp,
				Aligned: res.aligned, Reason: ReasonBacklogExceeded,
				Detail: fmt.Sprintf("rejected: pending at max %d (alignment pending)",
					e.cfg.Limits.MaxPendingObs)})
			return r
		}
		r.d = e.commit(Decision{Obs: o, Accepted: false, RawStamp: o.Stamp,
			Aligned: res.aligned, Reason: res.reason, Detail: res.detail})
		// 仍暂缓入缓冲（out_of_range 也保留，待映射恢复后重判）。
		r.p = pendingItem{obs: o, aligned: res.aligned, raw: o.Stamp, ready: false, kind: kind}
		r.pushed = e.buf.Push(r.p)
		return r
	}

	// 时钟漂移硬边界检查（对齐关闭时保留既有原始时间戳语义）。
	if !e.cfg.EnableAlignment && e.inited {
		drift := o.Stamp.Sub(e.est.Nominal.Stamp)
		if drift < 0 {
			drift = -drift
		}
		if drift > e.cfg.MaxClockDrift {
			r.d = e.commit(Decision{Obs: o, RawStamp: o.Stamp,
				Reason: ReasonTimeUnavailable,
				Detail: fmt.Sprintf("clock drift %v > max %v", drift, e.cfg.MaxClockDrift)})
			return r
		}
	}

	// 陈旧检查：以对齐后的统一时间与已融合时间比较。
	if e.inited && res.aligned.Before(e.est.Nominal.Stamp.Add(-e.cfg.StaleAfter)) {
		r.d = e.commit(Decision{Obs: o, RawStamp: o.Stamp,
			Aligned: res.aligned, Reason: ReasonStale,
			Detail: fmt.Sprintf("aligned %v older than state %v by > %v",
				res.aligned, e.est.Nominal.Stamp, e.cfg.StaleAfter)})
		return r
	}

	// 重复/倒退判定：同一路内以原始时间戳比较（对齐不改变单路时钟序）。
	if rr := e.buf.ClassifyRaw(kind, o.Stamp, o.Seq); rr == ReasonDuplicate || rr == ReasonTimestampRegress {
		r.d = e.commit(Decision{Obs: o, RawStamp: o.Stamp,
			Aligned: res.aligned, Reason: rr,
			Detail: "rejected by reorder buffer ordering"})
		return r
	}
	if e.buf.ContainsRaw(kind, o.Stamp, o.Seq) {
		r.d = e.commit(Decision{Obs: o, RawStamp: o.Stamp,
			Aligned: res.aligned, Reason: ReasonDuplicate,
			Detail: "identical observation already pending in buffer"})
		return r
	}

	r.p = pendingItem{obs: o, aligned: res.aligned, raw: o.Stamp, ready: true, kind: kind}
	if !e.buf.Push(r.p) {
		// 容量已在批前预检；到此视为积压拒绝。
		r.d = e.commit(Decision{Obs: o, RawStamp: o.Stamp,
			Aligned: res.aligned, Reason: ReasonBacklogExceeded,
			Detail: "reorder buffer full"})
		return r
	}
	r.pushed = true
	r.d = e.commit(Decision{Obs: o, Accepted: true, RawStamp: o.Stamp,
		Aligned: res.aligned, Reason: ReasonAccepted, Detail: "buffered"})
	return r
}

// retryPending 重判所有尚未就绪的待定观测（参考推进或模型转 locked
// 后它们可能变得可用）。
func (e *Engine) retryPending() {
	if !e.cfg.EnableAlignment {
		return
	}
	for i := 0; i < e.buf.Len(); i++ {
		p := e.buf.Index(i)
		if p.ready {
			continue
		}
		r, _ := e.align.Retry(p)
		if r.ready {
			p.ready = true
			p.aligned = r.aligned
			e.buf.SetIndex(i, p)
		}
	}
}

// drain 按全序消费缓冲中到期且对齐可信的观测（须持写锁）。
func (e *Engine) drain(deadline time.Time) {
	for {
		horizon, haveWM := e.alignedHorizon()
		idx, ok := e.buf.MinReadyIndex(horizon)
		if !ok {
			return
		}
		if !deadline.IsZero() && e.nowFunc().After(deadline) {
			return
		}
		_ = haveWM
		p, _ := e.buf.PopIndex(idx)
		e.consume(p)
	}
}

// alignedHorizon 计算本次可安全消费的统一时间上界：
// 对齐水线（各路覆盖交集）与乱序等待窗口取交集。
func (e *Engine) alignedHorizon() (time.Time, bool) {
	horizon := time.Time{}
	set := false
	if e.cfg.EnableAlignment {
		if wm, ok := e.align.watermark(); ok {
			horizon, set = wm, true
		}
	}
	if e.cfg.ReorderDelay > 0 {
		if maxReady, ok := e.buf.MaxReadyAligned(); ok {
			dh := maxReady.Add(-e.cfg.ReorderDelay)
			if !set || dh.Before(horizon) {
				horizon, set = dh, true
			}
		}
	}
	if !set {
		// 无任何水线限制：取远未来上界。
		return e.estOrFuture(), false
	}
	return horizon, true
}

// estOrFuture 提供不限制消费的上界（无水线时）。
func (e *Engine) estOrFuture() time.Time {
	if e.inited {
		return e.est.Nominal.Stamp.Add(100 * 365 * 24 * time.Hour)
	}
	if m, ok := e.buf.MaxReadyAligned(); ok {
		return m
	}
	return time.Time{}
}

// consume 消费一条已排序、已对齐的观测（须持写锁）。
func (e *Engine) consume(p pendingItem) {
	o := p.obs
	o.Stamp = p.aligned // 一切状态推进只用统一时间

	if !e.inited {
		e.initObs = append(e.initObs, o)
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
			e.consumed++
			e.commit(Decision{Obs: o, RawStamp: p.raw, Aligned: p.aligned,
				Reason: ReasonInsufficient,
				Detail: fmt.Sprintf("init window %d/%d (pose=%v)",
					len(e.initObs), e.cfg.InitWindow, hasPose)})
			return
		}
		e.initialize()
		e.consumed++
		e.commit(Decision{Obs: o, RawStamp: p.raw, Aligned: p.aligned,
			Accepted: true, Reason: ReasonAccepted, Detail: "initialized"})
		return
	}

	// 状态只能继续往后走：对齐换算后不晚于当前状态的观测不重复消费。
	dt := o.Stamp.Sub(e.est.Nominal.Stamp).Seconds()
	if dt > 0 {
		if o.Kind == SensorIMU {
			e.est.Predict(o, dt)
		} else {
			e.est.PredictConstantVelocity(dt)
		}
		e.est.AddProcessNoise(e.cfg.ProcessNoise, dt)
	}

	// 传感器降级检查：以各路统一时间衡量缺测间隔。
	e.refreshDegraded(o.Stamp)
	inflated := false
	if e.degraded[o.Kind] {
		inflated = true
		for i := range o.CovDiag {
			o.CovDiag[i] *= e.cfg.DegradedNoiseInflate
		}
	}

	switch o.Kind {
	case SensorIMU:
		// IMU 已在预测中消费；此处作为到达即接受的高频源。
		e.lastSrc[o.Kind] = o.Stamp
		e.consumed++
		e.commit(Decision{Obs: o, RawStamp: p.raw, Aligned: p.aligned,
			Accepted: true, Reason: ReasonAccepted})
		return
	case SensorOdometry:
		e.updateWith(o, p, inflated, e.est.UpdateOdometry)
	case SensorAbsolutePose:
		e.updateWith(o, p, inflated, e.est.UpdateAbsolutePose)
	default:
		e.commit(Decision{Obs: o, RawStamp: p.raw, Aligned: p.aligned,
			Reason: ReasonInsufficient, Detail: "unknown sensor kind"})
		return
	}
}

// updateWith 执行门限更新并记录可区分判定。
func (e *Engine) updateWith(o Observation, p pendingItem, inflated bool,
	upd func(Observation, float64) (float64, bool)) {
	d2, ok := upd(o, e.cfg.GateChi2)
	if !ok {
		e.rejected++
		e.commit(Decision{Obs: o, RawStamp: p.raw, Aligned: p.aligned,
			Reason: ReasonOutlierGate, Maha: d2,
			Detail: fmt.Sprintf("mahalanobis^2 %.3f > gate %.3f", d2, e.cfg.GateChi2)})
		return
	}
	e.lastSrc[o.Kind] = o.Stamp
	e.consumed++
	if inflated {
		e.commit(Decision{Obs: o, RawStamp: p.raw, Aligned: p.aligned,
			Accepted: true, Reason: ReasonSensorDegraded, Maha: d2,
			Detail: fmt.Sprintf("sensor %s degraded; noise inflated x%.0f",
				o.Kind, e.cfg.DegradedNoiseInflate)})
		return
	}
	e.commit(Decision{Obs: o, RawStamp: p.raw, Aligned: p.aligned,
		Accepted: true, Reason: ReasonAccepted, Maha: d2})
}

// initialize 用缓冲的初始观测建立初始状态（时间戳已统一）。
func (e *Engine) initialize() {
	var init State
	var cov [9]float64
	for i := range cov {
		cov[i] = 1.0
	}
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

// refreshDegraded 依据各路最近观测的统一时间刷新降级标记（须持写锁）。
func (e *Engine) refreshDegraded(now time.Time) {
	for k := SensorIMU; k <= SensorAbsolutePose; k++ {
		last, seen := e.lastSrc[k]
		if !seen {
			continue // 从未出现的路不判降级
		}
		e.degraded[k] = now.Sub(last) > e.cfg.DegradeAfter
	}
}

// AdvanceTo 将状态预测推进到统一时刻 t：先消费缓冲中不晚于 t 的
// 已对齐观测，再纯预测剩余区间。目标时刻不晚于当前状态时返回可区分错误。
func (e *Engine) AdvanceTo(t time.Time) Decision {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.inited && e.buf.Len() == 0 {
		return Decision{Reason: ReasonInsufficient, Detail: "not initialized"}
	}
	e.retryPending()
	e.buf.RemapAll(e.align)
	limit := t
	if wm, ok := e.alignedHorizon(); ok && wm.Before(limit) {
		limit = wm
	}
	for {
		idx, ok := e.buf.MinReadyIndex(limit)
		if !ok {
			break
		}
		p, _ := e.buf.PopIndex(idx)
		e.consume(p)
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

// AlignmentStats 返回各路在线时间对齐状态。
func (e *Engine) AlignmentStats() []AlignStat {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.align.Stats()
}

// AlignedTime 返回某路原始时间戳在统一时间轴上的换算结果。
// 参考路恒等；非参考路仅在对齐锁定时返回 (换算时间, true)，否则返回
// (零值, false)。
func (e *Engine) AlignedTime(kind SensorKind, raw time.Time) (time.Time, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.align.AlignedTime(kind, raw)
}

// Stats 返回运行期规模统计快照。
func (e *Engine) Stats() EngineStats {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return EngineStats{
		Pending:       e.buf.Len(),
		PendingHigh:   e.pendingHigh,
		Consumed:      e.consumed,
		Rejected:      e.rejected,
		MaxBatchSize:  e.cfg.Limits.MaxBatchSize,
		MaxPendingObs: e.cfg.Limits.MaxPendingObs,
	}
}

// Decisions 返回判定日志副本。
func (e *Engine) Decisions() []Decision {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]Decision, len(e.decisions))
	copy(out, e.decisions)
	return out
}

// commit 记录判定（有界滚动），并返回该判定。
func (e *Engine) commit(d Decision) Decision {
	if len(e.decisions) >= maxDecisionLog {
		e.decisions = e.decisions[1:]
	}
	e.decisions = append(e.decisions, d)
	return d
}
