package fusion

import (
	"math"
	"sort"
	"time"
)

// 在线时间对齐层设计：
//
// 统一时间轴锚定在“参考路”（默认 IMU）上，参考路恒等映射。对每一
// 条非参考路观测，利用各路共享的物理运动信号（机体系 z 角速度；缺
// 角速度时用姿态偏航角）在参考信号上做确定性时延搜索，得到
// (原始时间戳 → 统一时间) 锚点对；锚点对在滑动时间窗内做中心化
// 最小二乘，持续估计仿射映射 unified = skew·raw + offset，从而跟
// 上运行中变化的漂移。
//
// 时延搜索只依赖观测内容与单调参考信号，与到达顺序无关，因此乱序、
// 丢帧不会污染配对。零漂移输入上残差恒为零、映射恒等，行为不变。

// refRingSpan / refRingCap 参考路样本环的时间跨度与条数双界。
// 跨度需覆盖“对齐窗口 + 可接受的单路缺测间隔”，使长缺测恢复后
// 仍能用历史参考做相关，且末端不会因单点保留而丢失基准。
const (
	refRingSpan = 200 * time.Second
	refRingCap  = 40000
	// lagCoarseStep 时延粗搜索步长；lagFineStep 粗定位后的精搜索步长。
	lagCoarseStep = 10 * time.Millisecond
	lagFineStep   = 2 * time.Millisecond
	// skewCoarseStep / skewFineStep 时钟斜率联合搜索的粗/细步长。
	skewCoarseStep = 0.002
	skewFineStep   = 0.0004
	// skewTrackBand 锁定后斜率的窄跟踪带宽（相对当前估计）。
	skewTrackBand = 0.008
	// anchorStride 每路两次锚点估计之间的最小原始时间间隔。
	anchorStride     = 250 * time.Millisecond
	anchorEvalStride = 500 * time.Millisecond
	// lockedEvalStride 锁定且残差良好时的低频复核间隔（零漂移长跑近零
	// 开销；漂移出现会打回校准态并恢复高频评估）。
	lockedEvalStride = 2 * time.Second
)

// sigSample 带时间戳的运动信号原始样本。
//   - IMU/里程计：v 为机体系 z 角速度（rad/s），rate=true
//   - 绝对位姿：v 为缠绕在 (−π,π] 的偏航角，rate=false；连续偏航由
//     sortedYaw 读取时从有序环首全局解卷绕，免疫乱序与偶发缺帧。
type sigSample struct {
	tNS  int64
	v    float64
	rate bool
	cum  float64 // rate 型：从环首到该点的累积偏航角（O(log n) 读取）
}

// featSpan 保留以兼容二阶导相关实验（当前相关使用偏航角）。
const featSpan = 150 * time.Millisecond

// sigWindow 一个短时信号窗口（t 为相对窗口中心的偏移）。
type sigWindow struct {
	t []int64
	v []float64
}

// clockPair 一个 (原始时间, 统一时间) 锚点对，整数纳秒；sk 为联合
// 搜索得到的时钟斜率（unified=skew·raw+offset）。diag=true 表示该
// 锚点只用于“越界诊断”（在扩展斜率范围才对齐），不用于正常拟合。
type clockPair struct {
	x    int64
	y    int64
	sk   float64
	diag bool
}

// clockModel 单路时钟映射：unified = skew·raw + offset。
type clockModel struct {
	state    AlignState
	skew     float64
	offsetNS int64
	residNS  float64     // 近期窗口残差 RMS（ns）
	pairs    []clockPair // 近期锚点对（换算/残差窗口内）
	hist     []clockPair // 长历史锚点对（稳健估计斜率，跨度更长）
	count    int         // 累计锚点数
	sign     int         // 锁定后的偏移符号（+1/-1/0）

	lastAnch    int64       // 上次锚点尝试的原始时间（ns）
	hasLastAnch bool        // 是否已记录上次尝试
	anchoredAt  int64       // 最近一次真正采纳锚点的锚点时间（ns）
	lastAdopt   bool        // 上一次尝试是否采纳
	firstSeen   int64       // 首次见到该路观测的统一时间（ns，相对原点）
	hasFirst    bool        // 是否已记录 firstSeen
	sig         []sigSample // 该路近期信号窗口（原始时间坐标，有界）
	lastRaw     float64     // 上一次位姿原始（缠绕）偏航，用于解卷绕
}

// sigWindowSpan / sigWindowCap 每路信号窗的时间跨度与条数双界。
// 跨度须覆盖最大相关窗口（半宽 4s × 2）外加余量，否则窗口无法构建。
const (
	sigWindowSpan = 12 * time.Second
	sigWindowCap  = 1600
)

// Alignment 在线时间对齐层（状态机见 AlignState 注释）。
type Alignment struct {
	cfg       Config
	preferred SensorKind
	autoRef   bool
	hasPref   bool
	ref       SensorKind
	refSet    bool
	origin    time.Time
	originSet bool        // 原点只在首个参考样本到达时锚定一次
	refRing   []sigSample // 参考路相关信号环（同时承担时间覆盖信息）
	refYaw    float64     // 参考路累积偏航角（单调相关信号）
	refPrevT  int64
	refHasYaw bool
	models    map[SensorKind]*clockModel
	frontier  map[SensorKind]time.Time
	draining  bool // AdvanceTo 排空口径：允许半窗右外推，放出暂缓尾部
}

// NewAlignment 创建在线时间对齐层。
func NewAlignment(cfg Config) *Alignment {
	return &Alignment{
		cfg:       cfg,
		preferred: cfg.AlignRef,
		autoRef:   cfg.AlignAutoRef,
		hasPref:   !cfg.AlignAutoRef,
		models:    make(map[SensorKind]*clockModel),
		frontier:  make(map[SensorKind]time.Time),
	}
}

// IsEnabled 是否启用在线对齐。
func (a *Alignment) IsEnabled() bool { return a.cfg.EnableAlignment }

// Ref / isRefSet / SelectRef 参考路查询与选择。
func (a *Alignment) Ref() (SensorKind, bool) { return a.ref, a.refSet }
func (a *Alignment) isRefSet() bool          { return a.refSet }
func (a *Alignment) SelectRef(k SensorKind) {
	if a.refSet {
		return
	}
	a.refSet, a.ref = true, k
}

func (a *Alignment) relNS(t time.Time) int64 { return t.Sub(a.origin).Nanoseconds() }
func (a *Alignment) fromNS(ns int64) time.Time {
	return a.origin.Add(time.Duration(ns))
}

// obsRate 取观测用于相关的瞬时偏航角速度（IMU/里程计），绝对位姿
// 返回 0（其相关信号直接由姿态偏航提供）。
func obsRate(o Observation) float64 {
	switch o.Kind {
	case SensorIMU:
		return o.Gyro.Z
	case SensorOdometry:
		return o.AngVel.Z
	}
	return 0
}

// poseYaw 从绝对位姿四元数提取绕 z 的偏航角（范围 (−π,π]）。
func poseYaw(o Observation) float64 { return QuatToRotVec(o.Pose.Rotation).Z }

// unwrapYaw 把缠绕在 (−π,π] 的位姿偏航角累积为连续角（按该路有序
// 信号的最近一个角度样本取最短增量）。
func (m *clockModel) unwrapYaw(raw float64) float64 {
	if len(m.sig) == 0 {
		return raw
	}
	last := m.sig[len(m.sig)-1].v
	return last + wrapDelta(last, raw)
}

// wrapDelta 返回从缠绕角 a 到 b 的最短有向增量（(−π,π]）。
func wrapDelta(a, b float64) float64 {
	d := b - a
	for d > math.Pi {
		d -= 2 * math.Pi
	}
	for d < -math.Pi {
		d += 2 * math.Pi
	}
	return d
}

// pushRef 追加参考路原始角速度样本（时间有序）并按双界修剪。
func (a *Alignment) pushRef(o Observation) {
	ns := a.relNS(o.Stamp)
	insertSig(&a.refRing, sigSample{tNS: ns, v: obsRate(o), rate: true})
	span := int64(refRingSpan)
	if len(a.refRing) > 1 {
		last := a.refRing[len(a.refRing)-1].tNS
		trimmed := false
		for len(a.refRing) > 1 && (last-a.refRing[0].tNS > span || len(a.refRing) > refRingCap) {
			a.refRing = a.refRing[1:]
			trimmed = true
		}
		if trimmed {
			rebaseCum(a.refRing)
		}
	}
	if o.Stamp.After(a.frontier[a.ref]) {
		a.frontier[a.ref] = o.Stamp
	}
}

// insertSig 按 tNS 有序插入样本（同时刻覆盖）。rate 型环在插入后
// 维护累积偏航角 cum（从插入位置前一点起重算后缀），供 O(log n) 读取。
func insertSig(s *[]sigSample, q sigSample) {
	ss := *s
	i := sort.Search(len(ss), func(j int) bool { return ss[j].tNS >= q.tNS })
	if i < len(ss) && ss[i].tNS == q.tNS {
		ss[i].v, ss[i].rate = q.v, q.rate
	} else {
		ss = append(ss, sigSample{})
		copy(ss[i+1:], ss[i:])
		ss[i] = q
	}
	recomputeCum(ss, i)
	*s = ss
}

// recomputeCum 从 idx 之前一点起重算累积连续偏航角：rate 型做梯形
// 积分；角度型把缠绕角按有序前驱逐跳解卷绕。两种都从环首为 0 基准。
func recomputeCum(ss []sigSample, idx int) {
	if len(ss) == 0 {
		return
	}
	start := idx
	if start > 0 {
		start--
	}
	if start == 0 {
		ss[0].cum = ss[0].v
		if ss[0].rate {
			ss[0].cum = 0
		}
		start = 1
	}
	for k := start; k < len(ss); k++ {
		if ss[0].rate {
			dt := float64(ss[k].tNS-ss[k-1].tNS) / float64(time.Second)
			ss[k].cum = ss[k-1].cum + 0.5*(ss[k-1].v+ss[k].v)*dt
		} else {
			ss[k].cum = ss[k-1].cum + wrapDelta(ss[k-1].v, ss[k].v)
			// 注意：wrapDelta 的基准是“缠绕角”，cum 仅用于读取；为保证
			// 连续累积正确，解卷绕应基于上一点的缠绕角，这里 v 即缠绕角。
		}
	}
}

// rebaseCum 修剪环首后把累积连续偏航角平移到新环首基准（速率环
// 首点归零；角度环首点对齐其缠绕角）。
func rebaseCum(ss []sigSample) {
	if len(ss) == 0 {
		return
	}
	var base float64
	if ss[0].rate {
		base = ss[0].cum
	} else {
		base = ss[0].cum - ss[0].v
	}
	for i := range ss {
		ss[i].cum -= base
	}
}

// sortedYaw 在已按时间排序的信号环上返回 ns 处的连续（解卷绕/积分）
// 偏航角，统一基于预计算的 cum 字段，O(log n)。
func sortedYaw(s []sigSample, ns int64) (float64, bool) {
	if len(s) == 0 || ns < s[0].tNS || ns > s[len(s)-1].tNS {
		return 0, false
	}
	i := sort.Search(len(s), func(j int) bool { return s[j].tNS >= ns })
	if i == 0 {
		if s[0].rate {
			return 0, true
		}
		return s[0].cum, true
	}
	lo, hi := s[i-1], s[i]
	if hi.tNS == lo.tNS {
		return lo.cum, true
	}
	f := float64(ns-lo.tNS) / float64(hi.tNS-lo.tNS)
	if s[0].rate {
		// 段内按梯形积分补一小段。
		dt := float64(ns-lo.tNS) / float64(time.Second)
		return lo.cum + 0.5*(lo.v+hi.v)*dt, true
	}
	// 角度型：在两个连续 cum 值之间线性插值。
	return lo.cum + f*(hi.cum-lo.cum), true
}

func (a *Alignment) model(k SensorKind) *clockModel {
	m, ok := a.models[k]
	if !ok {
		m = &clockModel{state: AlignCold, skew: 1}
		a.models[k] = m
	}
	return m
}

// refAt 返回参考连续偏航角在统一时间 ns 处的值（由有序速率积分）。
func (a *Alignment) refAt(ns int64) (float64, bool) {
	return sortedYaw(a.refRing, ns)
}

// refCovers 判断参考信号是否完整覆盖统一时间区间 [lo,hi]。
func (a *Alignment) refCovers(lo, hi int64) bool {
	if len(a.refRing) == 0 {
		return false
	}
	return lo >= a.refRing[0].tNS && hi <= a.refRing[len(a.refRing)-1].tNS
}

// pushSig 按时间戳有序插入一条该路原始信号（IMU/odom 为角速度、
// 绝对位姿为解卷绕偏航角），批内乱序到达不破坏单调性，并按双界修剪。
func (a *Alignment) pushSig(m *clockModel, ns int64, v float64, rate bool) {
	insertSig(&m.sig, sigSample{tNS: ns, v: v, rate: rate})
	span := int64(sigWindowSpan)
	if len(m.sig) > 1 {
		last := m.sig[len(m.sig)-1].tNS
		trimmed := false
		for len(m.sig) > 2 && (last-m.sig[0].tNS > span || len(m.sig) > sigWindowCap) {
			m.sig = m.sig[1:]
			trimmed = true
		}
		if trimmed {
			rebaseCum(m.sig)
		}
	}
}

// 相关窗口的默认半宽与采样步长。窗口需覆盖足够的速率变化，使单调
// 偏航角的整窗曲线可唯一确定平移（短窗在近匀速段会线性退化）。
const (
	sigHalfWin = 2500 * time.Millisecond
	sigWinStep = 150 * time.Millisecond
)

// buildSigWindow 以 ns 为中心，从该路信号环中取覆盖 [−half,+half]
// 的等间隔样本（参考侧在 refAt 中插值，因此步长可自由选）。
// 用当前时刻之前为主、之后尽量取（到达即估时后续点可能不足，由
// windowCost 的覆盖判定拒绝无效候选）。
// halfWinFor 各路相关窗口半宽。低速率绝对位姿路用较短窗口（其信号
// 干净、5Hz 即可辨识）；高速率里程计路需更长窗口以克服近匀速段的
// 线性退化。窗口延迟仅半宽，兼顾在线性与辨识度。
func (a *Alignment) halfWinFor(k SensorKind) time.Duration {
	if k == SensorAbsolutePose {
		return 2500 * time.Millisecond
	}
	return sigHalfWin
}

// buildSigWindow 以 ns 为中心，从该路信号环取覆盖 [−half,+half]
// 的等间隔样本（参考侧插值，步长可自由选）。
func buildSigWindow(m *clockModel, ns, half int64) (sigWindow, bool) {
	if len(m.sig) < 5 {
		return sigWindow{}, false
	}
	var w sigWindow
	step := int64(sigWinStep)
	for dt := -half; dt <= half; dt += step {
		v, ok := sigAt(m.sig, ns+dt)
		if !ok {
			continue
		}
		w.t = append(w.t, dt)
		w.v = append(w.v, v)
	}
	return w, len(w.t) >= 13
}

// sigAt 返回该路连续偏航角在 ns 处的值（速率环积分 / 角度环插值）。
func sigAt(s []sigSample, ns int64) (float64, bool) {
	return sortedYaw(s, ns)
}

// windowCostSkew 在给定时钟斜率 skew（unified = skew·raw + offset）
// 下计算形状残差：观测窗口的原始时间偏移 dt 对应统一时间 skew·dt，
// 再加 lag 平移到参考轴，随后两侧去均值比较形状（常数偏置无关）。
func (a *Alignment) windowCostSkew(centerNS, lag int64, skew float64, win sigWindow) (float64, bool) {
	rs := make([]float64, len(win.t))
	var sw, sr float64
	for i, dt := range win.t {
		uni := centerNS + int64(math.Round(float64(dt)*skew)) + lag
		rv, ok := a.refAt(uni)
		if !ok {
			return 0, false
		}
		rs[i] = rv
		sw += win.v[i]
		sr += rv
	}
	n := float64(len(win.t))
	mw, mr := sw/n, sr/n
	var ss float64
	for i := range win.t {
		d := (rs[i] - mr) - (win.v[i] - mw)
		ss += d * d
	}
	return ss, true
}

// rateSpreadGate 触发时间对齐搜索所需的窗口角速度变化范围（rad/s）。
// 低于它视为近似匀速段：偏航呈周期性，去均值形状仍有整周期假峰，斜
// 率/偏移不可辨识，故跳过锚点、保持恒等（零漂移兼容、零额外开销）。
const rateSpreadGate = 0.10

// windowRateSpread 返回窗口内角速度的变化范围（max−min, rad/s）。
// 速率用约 0.6s 的长间隔差分估计（而非相邻点），把高频角速度/姿态噪
// 声平均掉：近似匀速段速率估计稳定（极差仅为噪声残差），时变速率段
// 速率在明显范围内变化。用于判断“这段运动能否辨识时钟斜率/偏移”。
func (a *Alignment) windowRateSpread(win sigWindow) float64 {
	if len(win.t) < 4 {
		return 0
	}
	const rateDT = int64(600 * time.Millisecond)
	mn, mx := math.Inf(1), math.Inf(-1)
	n := 0
	// win.t 等间隔（sigWinStep），用相隔 stride 个点的差分得到 ~0.6s。
	stride := int(rateDT / int64(sigWinStep))
	if stride < 1 {
		stride = 1
	}
	for i := stride; i < len(win.v); i++ {
		dt := float64(win.t[i]-win.t[i-stride]) / float64(time.Second)
		if dt <= 0 {
			continue
		}
		r := (win.v[i] - win.v[i-stride]) / dt
		if r < mn {
			mn = r
		}
		if r > mx {
			mx = r
		}
		n++
	}
	if n == 0 || math.IsInf(mn, 1) {
		return 0
	}
	return mx - mn
}

// lagSearchSkew 在给定 skew 下两级搜索最佳 lag，返回 (lag, 残差和, 命中)。
func (a *Alignment) lagSearchSkew(win sigWindow, centerNS int64, skew, predLag, radius, acceptMeanSq float64) (int64, float64, bool) {
	pred := int64(math.Round(predLag))
	rad := int64(radius)
	best, bestCost := pred, math.Inf(1)
	consider := func(lag int64) {
		c, ok := a.windowCostSkew(centerNS, lag, skew, win)
		if ok && (c < bestCost || (c == bestCost && abs64(lag) < abs64(best))) {
			best, bestCost = lag, c
		}
	}
	coarse := int64(lagCoarseStep)
	for lag := pred - rad; lag <= pred+rad; lag += coarse {
		consider(lag)
	}
	lo, hi := best-coarse, best+coarse
	if lo < pred-rad {
		lo = pred - rad
	}
	if hi > pred+rad {
		hi = pred + rad
	}
	for lag := lo; lag <= hi; lag += int64(lagFineStep) {
		consider(lag)
	}
	return best, bestCost, bestCost/float64(len(win.t)) <= acceptMeanSq
}

// admitResult / 对外判定。
type admitResult struct {
	ready   bool
	reason  RejectReason
	aligned time.Time
	detail  string
	ops     int // 本次判定的折算操作消耗（时延搜索等重操作）
}

// Admit 摄入一条观测：参考路入信号环；非参考路按状态机更新时钟模型
// 并给出对齐判定。
func (a *Alignment) Admit(o Observation) admitResult {
	if !a.cfg.EnableAlignment {
		return admitResult{ready: true, aligned: o.Stamp}
	}
	if !a.originSet {
		// 原点在对齐层首次见到任意观测时即锚定，避免非参考路在参考
		// 到达前以零原点产生天文坐标、污染信号环。它只是数值坐标原点。
		a.origin = o.Stamp
		a.originSet = true
	}
	if !a.refSet {
		switch {
		case a.hasPref && o.Kind == a.preferred:
			a.SelectRef(o.Kind)
		case a.autoRef && o.Kind == SensorIMU:
			a.SelectRef(SensorIMU) // 自动模式：IMU 优先
		case a.autoRef:
			a.SelectRef(o.Kind) // 自动模式且尚无 IMU：首达路兜底
		default:
			// 等待显式配置的参考路到达；观测暂缓，不丢弃。
			return admitResult{ready: false, reason: ReasonAlignCalibrating,
				aligned: o.Stamp, detail: "waiting for configured timebase reference"}
		}
	}
	if o.Kind == a.ref {
		a.pushRef(o)
		return admitResult{ready: true, aligned: o.Stamp}
	}

	m := a.model(o.Kind)
	if t := a.frontier[o.Kind]; o.Stamp.After(t) {
		a.frontier[o.Kind] = o.Stamp
	}
	ns := a.relNS(o.Stamp)
	// 记录该路首次被见到时的“参考统一时间”，用于校准宽限判定（避免
	// 长期不可信时观测无限挂起、饿死融合）。
	if !m.hasFirst {
		m.hasFirst = true
		if len(a.refRing) > 0 {
			m.firstSeen = a.refRing[len(a.refRing)-1].tNS
		} else {
			m.firstSeen = ns
		}
	}
	// 缓存原始信号：IMU/odom 存角速度（rate），绝对位姿存解卷绕偏航
	// 角。连续偏航角在相关读取时由有序环统一积分/插值得到，免疫乱序。
	if o.Kind == SensorAbsolutePose {
		// 存缠绕偏航角，连续化在 sortedYaw 读取时统一完成。
		a.pushSig(m, ns, poseYaw(o), false)
	} else {
		a.pushSig(m, ns, obsRate(o), true)
	}
	ops := 1

	// 锚点为“居中窗口”。为保证在该路时钟存在正偏移（原始戳超前统一
	// 时间）时参考侧也已覆盖窗口远端，锚点相对到达延迟
	// half+AlignSearchRange；此时无论真实偏移在搜索域内何处，统一时
	// 间上的参考数据都已到齐。条件不足时 tryAnchorAt 返回 skipped。
	half := int64(a.halfWinFor(o.Kind))
	anchorNS := ns - half - int64(a.cfg.AlignSearchRange)
	// 锚点评估限频：校准期按 anchorEvalStride；一旦锁定且残差良好（零/
	// 小漂移恒成立），按更长的 lockedEvalStride 低频复核即可，这使零
	// 漂移长跑的相关开销接近于零；残差升高（漂移出现/变化）时 refit
	// 会把状态打回 calibrating，自动恢复高频评估。
	evalStride := int64(anchorEvalStride)
	if m.state == AlignLocked && m.residNS <= float64(a.cfg.AlignTolerance) {
		evalStride = int64(lockedEvalStride)
	}
	if !m.hasLastAnch || anchorNS-m.lastAnch >= evalStride {
		used, outcome := a.tryAnchorAt(m, o.Kind, anchorNS, half)
		ops = used
		if outcome != anchorSkipped {
			m.lastAnch = anchorNS
			m.hasLastAnch = true
			m.lastAdopt = outcome == anchorAdopted
		}
	}
	r := a.verdict(m, o, ns)
	r.ops = ops
	return r
}

// 搜索半径：冷启动需覆盖最大初始固定偏差（AlignDetect~AlignMaxDrift），
// 故取整个可信漂移范围；锁定后两次锚点间漂移变化很小，但为稳健仍在
// 同一范围内相关（窗口形状唯一，不会产生跨周期假峰）。平局偏好小 lag。
func (a *Alignment) searchRadius(m *clockModel) float64 {
	return float64(a.cfg.AlignSearchRange)
}

// anchorStrideFor 返回各路两次“已采纳”锚点间的最小原始时间间隔。
func (a *Alignment) anchorStrideFor(k SensorKind) time.Duration {
	if k == SensorAbsolutePose {
		return 250 * time.Millisecond
	}
	return anchorStride
}

// retryStrideFor 上次锚点未采纳时的重试间隔；必须短于该路采样周期，
// 否则会因“每次推进量 < 门限”而被永久卡住。
func (a *Alignment) retryStrideFor(k SensorKind) time.Duration {
	if k == SensorAbsolutePose {
		return 100 * time.Millisecond // pose 5Hz → 周期 200ms
	}
	return 40 * time.Millisecond // odom 20Hz → 周期 50ms
}

// anchorOutcome 描述一次锚点尝试的结果。
type anchorOutcome int

const (
	anchorSkipped anchorOutcome = iota // 条件不足，未真正搜索（不应消耗尝试机会）
	anchorMissed                       // 已搜索但残差不达标
	anchorAdopted                      // 已采纳为锚点
)

// tryAnchorAt 在给定锚点时刻 anchorNS（该路原始坐标）用居中信号
// 窗口做时延相关。返回 (折算操作消耗, 结果)。
func (a *Alignment) tryAnchorAt(m *clockModel, kind SensorKind, anchorNS, half int64) (int, anchorOutcome) {
	if len(a.refRing) < 8 {
		return 1, anchorSkipped
	}
	// 预测当前 lag：冷启动为 0；已锁定时必须用“锚点分段换算”预测，而不是
	// 全局仿射 applyModel。全局仿射 unified=skew·raw+offset 的 skew/offset
	// 是相对同步原点拟合的，在远离原点（raw 很大）时 (skew−1)·raw 项会把
	// 预测 lag 放大到数秒（即便真实当前 lag 接近 0），把搜索中心带偏到可
	// 行域边缘的假峰；分段锚点直接给出当前时刻附近的 lag(y−x)，是正确、
	// 有界的预测。
	predLag := int64(0)
	if m.state == AlignLocked {
		if y, ok := anchorsInterp(m, m.hist, anchorNS); ok {
			predLag = y - anchorNS
		}
	}
	// 可行 lag 区间（使窗口整体落在参考覆盖 [refLo,refHi] 内）：
	// refLo ≤ anchorNS+lag−half 且 anchorNS+lag+half ≤ refHi
	// ⇒ lag ∈ [refLo−anchorNS+half, refHi−anchorNS−half]。
	halfWin := half
	refLo := a.refRing[0].tNS
	refHi := a.refRing[len(a.refRing)-1].tNS
	feasLo := refLo - anchorNS + halfWin
	feasHi := refHi - anchorNS - halfWin
	rad := int64(a.cfg.AlignSearchRange)
	if feasHi < -rad || feasLo > rad {
		return 1, anchorSkipped
	}
	win, ok := buildSigWindow(m, anchorNS, half)
	if !ok {
		return 1, anchorSkipped
	}
	// 该路窗口自身的净转角：无转动则相关不可辨识，直接跳过（检查在
	// 该路自身信号上做，与坐标系无关，不受漂移影响）。
	if math.Abs(win.v[len(win.v)-1]-win.v[0]) < 0.05 {
		return 1, anchorSkipped
	}
	// 联合 (skew, lag) 搜索。为控制成本：
	//  1. skew 用粗网格（skewCoarseStep）只做一次粗 lag 相关，定位最优
	//     斜率（形状拉伸对斜率敏感，粗步长已能区分）；
	//  2. 在最优斜率附近再做一次细 skew（skewFineStep）与两级 lag 精化。
	// skew 搜索域以已锁定斜率为中心、钳制在可信范围。
	trustSq := 0.012 * 0.012
	if kind == SensorAbsolutePose {
		trustSq = 0.02 * 0.02
	}
	radF := a.searchRadius(m)

	// 时变速率门控：只有窗口内速率变化范围足够，时间偏移/斜率才可辨识。
	// 近似匀速段（恒定圆周）偏航呈周期性，存在整周期假峰，无法区分时钟
	// 偏移，此时跳过锚点、保持恒等（零漂移兼容、零额外开销）。
	if a.windowRateSpread(win) < rateSpreadGate {
		return 1, anchorSkipped
	}

	skewCenter := 1.0
	if m.state == AlignLocked && m.skew > 0 {
		skewCenter = m.skew
	}
	coarseOnly := func(sk float64) (int64, float64) {
		return a.lagCoarse(win, anchorNS, sk, float64(predLag), radF)
	}
	bestSkew, bestLag, bestCost := 1.0, int64(0), math.Inf(1)
	probes := 0
	// 斜率搜索域：冷启动在整个可信范围（±AlignMaxSkew）宽扫一次；
	// 锁定后只在当前斜率附近窄跟踪（±skewTrackBand），大幅降低成本，
	// 同时足以跟上运行中漂移斜率的渐进/分段变化。
	skewLo, skewHi := 1-a.cfg.AlignMaxSkew, 1+a.cfg.AlignMaxSkew
	if m.state == AlignLocked && m.skew > 0 {
		b := skewTrackBand
		skewLo, skewHi = skewCenter-b, skewCenter+b
		if skewLo < 1-a.cfg.AlignMaxSkew {
			skewLo = 1 - a.cfg.AlignMaxSkew
		}
		if skewHi > 1+a.cfg.AlignMaxSkew {
			skewHi = 1 + a.cfg.AlignMaxSkew
		}
	}
	for sk := skewLo; sk <= skewHi+1e-12; sk += skewCoarseStep {
		lag, c := coarseOnly(sk)
		if c < bestCost {
			bestCost, bestSkew, bestLag = c, sk, lag
		}
	}
	// 最优斜率附近细网格精化（含两级 lag）。
	skLo := bestSkew - skewCoarseStep
	skHi := bestSkew + skewCoarseStep
	for sk := skLo; sk <= skHi+1e-12; sk += skewFineStep {
		lag, c, _ := a.lagSearchSkew(win, anchorNS, sk, float64(predLag), radF, trustSq)
		if c < bestCost {
			bestCost, bestSkew, bestLag = c, sk, lag
		}
	}
	// 边缘余量：最优组合对应的统一时间窗口需落在参考覆盖内部。
	margin := int64(300 * time.Millisecond)
	cc := anchorNS + bestLag
	edgeOK := cc-half >= a.refRing[0].tNS+margin &&
		cc+half <= a.refRing[len(a.refRing)-1].tNS-margin
	if bestCost/float64(len(win.t)) <= trustSq && edgeOK {
		p := clockPair{x: anchorNS, y: anchorNS + bestLag, sk: bestSkew}
		m.pairs = append(m.pairs, p)
		m.hist = append(m.hist, p)
		m.count++
		m.anchoredAt = anchorNS
		a.trimWindow(m)
		a.refit(m)
		return probes, anchorAdopted
	}
	// 无论受限搜索是否采纳，校准阶段都在更宽的诊断域做一次粗扫，比较
	// “受限域最优残差”与“扩展域最优残差”：若扩展域（斜率越界）显著
	// 更贴合（残差不到受限域的一半），记录诊断锚点，由 refit 判定
	// out_of_range；这能发现“平移在受限域勉强对齐、但真实斜率越界”
	// 的情形。锁定后跳过（斜率由窄跟踪负责）。
	if m.state != AlignLocked {
		limited := bestCost
		if dsk, dlag, dc, dok := a.diagSkew(win, anchorNS, radF, trustSq); dok &&
			math.Abs(dsk-1) > a.cfg.AlignMaxSkew &&
			dc/float64(len(win.t)) <= trustSq &&
			dc < 0.5*limited {
			p := clockPair{x: anchorNS, y: anchorNS + dlag, sk: dsk, diag: true}
			m.hist = append(m.hist, p)
			a.trimWindow(m)
			a.refit(m)
		}
	}
	return probes, anchorMissed
}

// diagSkew 在“诊断域”（可信范围再外扩一个 AlignMaxSkew）做粗扫，
// 用于区分“斜率略超可信范围”(out_of_range) 与“完全无法对齐”
// (calibrating)。超出诊断域的极端时钟不会在此被采纳，保持校准态。
func (a *Alignment) diagSkew(win sigWindow, anchorNS int64, rad, trustSq float64) (float64, int64, float64, bool) {
	bestSkew, bestLag, bestCost := 1.0, int64(0), math.Inf(1)
	band := 2 * a.cfg.AlignMaxSkew
	for sk := 1 - band; sk <= 1+band+1e-12; sk += skewCoarseStep {
		lag, c := a.lagCoarse(win, anchorNS, sk, 0, rad)
		if c < bestCost {
			bestCost, bestSkew, bestLag = c, sk, lag
		}
	}
	return bestSkew, bestLag, bestCost, bestCost < math.Inf(1)
}

// lagCoarse 只在粗 lag 网格上评估（供联合搜索的斜率粗扫描使用）。
func (a *Alignment) lagCoarse(win sigWindow, centerNS int64, skew, predLag, radius float64) (int64, float64) {
	pred := int64(math.Round(predLag))
	rad := int64(radius)
	best, bestCost := pred, math.Inf(1)
	for lag := pred - rad; lag <= pred+rad; lag += int64(lagCoarseStep) {
		c, ok := a.windowCostSkew(centerNS, lag, skew, win)
		if ok && (c < bestCost || (c == bestCost && abs64(lag) < abs64(best))) {
			best, bestCost = lag, c
		}
	}
	return best, bestCost
}

// mapRawNS 把某路原始相对时间 x 换算到统一时间轴，返回 (统一时间,
// 是否可按已对齐/恒等身份采用)。所有对外口径——判定、待定项重算、
// 对齐水线、AlignedTime——必须走这里，确保“显示用什么时间、融合就
// 真的用什么时间”，避免同一观测在不同批次被重算成不同时间。
//
//   - locked：用近期确认锚点分段换算（局部斜率，能跟时变漂移）。向左
//     （早于最早锚点，即锁定瞬间被暂缓的启动段）与向右（晚于最新锚点的
//     当前前沿，稳态常态）都允许半窗量级短距外推；锚点持续到达，稳态右
//     外推可靠。启动段向左的正确性由“锁定前该路数据暂缓、水线压住参考”
//     保证——锁定瞬间这些点统一时间一致、用全局稳健仿射换算即可，不会
//     出现跨数百毫秒的错位（时钟在同步原点一致、斜率接近 1）；
//   - cold：未发现漂移，严格恒等（零漂移行为与关闭对齐逐位一致）；
//   - calibrating / out_of_range：不可采用，由调用方决定暂缓/拒绝。
func (a *Alignment) mapRawNS(m *clockModel, x int64) (int64, bool) {
	switch m.state {
	case AlignLocked:
		if y, ok := anchorsInterp(m, m.hist, x); ok {
			return y, true
		}
		return a.applyModel(m, x), true
	case AlignCold:
		return x, true
	default:
		return 0, false
	}
}

// mapStrictNS 是“在线消费”口径：除 mapRawNS 的规则外，locked 路只放行
// 原始时间不晚于“最新已采纳锚点 + 半窗”的观测。
//
// 锚点之间的插值由已采纳（不可变）锚点唯一确定，换算随后续数据不会再变；
// 最新锚点之后只允许半窗（相关居中窗宽度）短距右外推，误差有界（真实斜率
// 偏离 1 很小，半窗外推仅十几毫秒）且下个锚点很快把它夹住。更晚的点要靠
// 段斜率长距外推，该斜率每来新锚点都可能变——若这种点已被消费，新锚点会
// 让同一观测换算时间回跳（负 dt、把未来速度/位姿配到过去姿态），对高频
// 速度更新经 IMU 积分正反馈放大、在漂移换向点尤其致命。故更晚的点在线暂
// 缓；离线追平由 AdvanceTo 的排空口径（允许更大右外推）处理。
func (a *Alignment) mapStrictNS(m *clockModel, x int64) (int64, bool) {
	if m.state == AlignLocked {
		if _, hi, cov := a.anchorCoverage(m); cov {
			slack := int64(a.halfWinFor(a.strictKind(m)))
			if x > hi+slack {
				return 0, false
			}
		}
	}
	return a.mapRawNS(m, x)
}

// strictKind 从模型反查路别（仅用于按路选相关半窗）。模型数量很少，线性
// 扫描即可。
func (a *Alignment) strictKind(m *clockModel) SensorKind {
	for k := SensorIMU; k <= SensorAbsolutePose; k++ {
		if a.models[k] == m {
			return k
		}
	}
	return SensorOdometry
}

// anchorCoverage 返回该路正常（非诊断）已采纳锚点在原始时间轴上的覆盖
// 区间 [lo,hi]。hist 按追加顺序保存，正常点未必严格有序，取最小/最大。
func (a *Alignment) anchorCoverage(m *clockModel) (int64, int64, bool) {
	lo, hi, set := int64(0), int64(0), false
	for _, p := range m.hist {
		if p.diag {
			continue
		}
		if !set {
			lo, hi, set = p.x, p.x, true
			continue
		}
		if p.x < lo {
			lo = p.x
		}
		if p.x > hi {
			hi = p.x
		}
	}
	return lo, hi, set
}

// mapForConsume 选择消费口径：在线摄入（draining=false）走严格口径
// （只放出已被锚点夹住、换算永不回跳的点）；AdvanceTo 排空（draining=true）
// 走宽松口径（允许半窗右外推，把暂缓尾部一次放出）。
func (a *Alignment) mapForConsume(m *clockModel, x int64) (int64, bool) {
	if a.draining {
		return a.mapRawNS(m, x)
	}
	return a.mapStrictNS(m, x)
}

// SetDraining 设置/解除排空口径（由引擎在 AdvanceTo 排空前后切换）。
func (a *Alignment) SetDraining(on bool) { a.draining = on }

// verdict 依据模型状态给出对齐判定。
func (a *Alignment) verdict(m *clockModel, o Observation, ns int64) admitResult {
	if y, ok := a.mapForConsume(m, ns); ok {
		if m.state == AlignCold {
			// 未发现漂移：恒等采用，与关闭对齐完全一致。
			return admitResult{ready: true, aligned: o.Stamp, detail: "identity: no drift detected"}
		}
		return admitResult{ready: true, aligned: a.fromNS(y), detail: "aligned"}
	}
	if m.state == AlignOutOfRange {
		return admitResult{ready: false, reason: ReasonTimeUnavailable,
			aligned: o.Stamp, detail: "clock mapping skew/offset outside trusted range"}
	}
	// 校准宽限降级：宽限按“当前参考统一时间 − 该路首次出现时刻”
	// 计算（用观测自身时间会让旧挂起项永远落在宽限内、无法恢复）。
	// 宽限内暂不采用（原因明确为样本不足）；超过宽限仍未锁定则按原始
	// 时间恒等放行并显式标记为“未对齐降级”，避免饿死融合；只有真正
	// 越界才持续拒绝。
	grace := int64(a.cfg.AlignCalibrateAfter)
	var now int64
	if len(a.refRing) > 0 {
		now = a.refRing[len(a.refRing)-1].tNS
	} else {
		now = ns
	}
	if !m.hasFirst || now-m.firstSeen <= grace {
		return admitResult{ready: false, reason: ReasonAlignCalibrating,
			aligned: o.Stamp, detail: "collecting time-alignment anchors"}
	}
	return admitResult{ready: true, aligned: o.Stamp, reason: ReasonAlignDegraded,
		detail: "alignment untrusted after grace: identity fallback"}
}

// DegradedSensors 返回因对齐不可信而处于“恒等降级”的路集合（已过
// 校准宽限但仍未锁定、且未判越界）。
func (a *Alignment) DegradedSensors() map[SensorKind]bool {
	out := make(map[SensorKind]bool)
	for k, m := range a.models {
		if a.refSet && k == a.ref {
			continue
		}
		if m.state == AlignOutOfRange || m.state == AlignLocked {
			continue
		}
		if m.hasFirst {
			last := int64(0)
			if len(a.refRing) > 0 {
				last = a.refRing[len(a.refRing)-1].tNS
			}
			if last-m.firstSeen > int64(a.cfg.AlignCalibrateAfter) {
				out[k] = true
			}
		}
	}
	return out
}

// Retry 在参考路推进或模型更新后重判一条待定观测的就绪状态。锚点
// 由新到达的该路信号驱动形成，这里只按当前模型重新判定，不重复搜索。
func (a *Alignment) Retry(p pendingItem) (admitResult, bool) {
	if !a.cfg.EnableAlignment || p.kind == a.ref {
		return admitResult{ready: true, aligned: p.raw}, true
	}
	m, ok := a.models[p.kind]
	if !ok {
		return admitResult{ready: false, reason: ReasonAlignCalibrating,
			aligned: p.raw, detail: "no time-alignment model yet"}, false
	}
	ns := a.relNS(p.obs.Stamp)
	r := a.verdict(m, p.obs, ns)
	return r, r.ready
}

// Remap 用当前模型重算待定项的对齐时间与就绪状态。口径与 verdict
// 完全一致（都走 mapForConsume），保证同一条观测在“暂缓重判”和“缓冲
// 重算”两条路径上得到相同的统一时间，不会每批跳动；校准宽限后的恒等
// 降级放行也在此一致生效（带 alignFallback 标记）。
func (a *Alignment) Remap(p pendingItem) pendingItem {
	if !a.cfg.EnableAlignment || p.kind == a.ref {
		p.aligned, p.ready = p.raw, true
		return p
	}
	m := a.models[p.kind]
	if m == nil {
		// 尚未建立任何模型：按恒等采用（零漂移兼容），绝不挂起。
		p.aligned, p.ready = p.raw, true
		return p
	}
	// 直接复用 verdict：同一观测在“暂缓重判”与“缓冲重算”两条路径上的
	// 就绪/换算/降级口径必须完全一致。
	r := a.verdict(m, p.obs, a.relNS(p.raw))
	if r.ready {
		p.ready = true
		p.aligned = r.aligned
		p.alignFallback = r.reason == ReasonAlignDegraded
	} else {
		p.ready = false
		p.aligned = p.raw
		p.alignFallback = false
	}
	return p
}

func (a *Alignment) applyModel(m *clockModel, x int64) int64 {
	return int64(math.Round(m.skew*float64(x) + float64(m.offsetNS)))
}

// AlignedTime 见 Engine.AlignedTime。
func (a *Alignment) AlignedTime(kind SensorKind, raw time.Time) (time.Time, bool) {
	if !a.cfg.EnableAlignment {
		return raw, true
	}
	if a.refSet && kind == a.ref {
		return raw, true
	}
	m, ok := a.models[kind]
	if !ok {
		// 从未建立模型：恒等（零漂移兼容）。
		return raw, true
	}
	x := a.relNS(raw)
	switch m.state {
	case AlignLocked:
		if y, ok := anchorsInterp(m, m.hist, x); ok {
			return a.fromNS(y), true
		}
		return a.fromNS(a.applyModel(m, x)), true
	case AlignCold:
		// 未发现漂移：恒等。
		return raw, true
	default:
		// calibrating / out_of_range：不可信，不提供换算。
		return time.Time{}, false
	}
}

// anchorsInterp 在已按 x 排序的锚点上做分段线性换算，返回统一时间。
// 分段斜率只钳到以 1 为中心的紧绝对带（剔除异常锚点极端斜率），不
// 依赖滞后的全局 skew，因此能即时跟上漂移换向；跨度内插值，跨度外用最
// 近一段短距外推。
func anchorsInterp(m *clockModel, ps0 []clockPair, x int64) (int64, bool) {
	// 仅用正常采纳的锚点（排除越界诊断锚点），且按 x 排序。
	ps := make([]clockPair, 0, len(ps0))
	for _, p := range ps0 {
		if !p.diag {
			ps = append(ps, p)
		}
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].x < ps[j].x })
	seg := func(a, b clockPair) float64 {
		s := segSlope(a, b)
		// 钳到“以 1 为中心的紧绝对带”[1±segSlopeAbsBand]：带宽略大于锚点
		// 相关噪声造成的段斜率抖动，用于剔除异常锚点的极端段斜率；它不跟随
		// 滞后的全局稳健斜率 m.skew——漂移换向时 m.skew 仍反映换向前段，若
		// 以它为中心钳制会强制局部段映射偏向错误方向并正反馈发散。固定以 1
		// 为中心，真实的 ±0.3% 段斜率可自由通过，又能即时跟上换向。
		lo, hi := 1-segSlopeAbsBand, 1+segSlopeAbsBand
		if s < lo {
			s = lo
		}
		if s > hi {
			s = hi
		}
		return s
	}
	if len(ps) == 0 {
		return 0, false
	}
	if len(ps) == 1 {
		// 单锚点：锚点之外一律保持当前 lag（斜率 1），不凭单锚点斜率外推。
		return ps[0].y + (x - ps[0].x), true
	}
	if x <= ps[0].x {
		s := seg(ps[0], ps[1])
		return int64(math.Round(float64(ps[0].y) + s*float64(x-ps[0].x))), true
	}
	last := ps[len(ps)-1]
	if x >= last.x {
		s := seg(ps[len(ps)-2], last)
		return int64(math.Round(float64(last.y) + s*float64(x-last.x))), true
	}
	i := sort.Search(len(ps)-1, func(j int) bool { return ps[j+1].x >= x })
	lo, hi := ps[i], ps[i+1]
	s := seg(lo, hi)
	return int64(math.Round(float64(lo.y) + s*float64(x-lo.x))), true
}

// segSlopeAbsBand 分段斜率相对 1 的绝对钳制带宽：略大于相关噪声造成的
// 段斜率抖动（真实测试漂移斜率在 ±0.3% 内），用于剔除异常锚点极端斜率，
// 又不依赖滞后的全局斜率，保证换向时局部段可自由取到真实斜率。
const segSlopeAbsBand = 0.01

func segSlope(a, b clockPair) float64 {
	dx := float64(b.x - a.x)
	if dx == 0 {
		return 1
	}
	return float64(b.y-a.y) / dx
}

// trimWindow 保留时间窗内锚点，并始终留一个滚动锚（窗口内最早点）。
func trimPairs(p []clockPair, span int64, keepOne bool) []clockPair {
	if len(p) < 2 || span <= 0 {
		return p
	}
	maxX := p[len(p)-1].x
	cut := 0
	for cut < len(p)-1 && maxX-p[cut].x > span {
		cut++
	}
	if cut > 0 {
		start := cut
		if keepOne {
			start = cut - 1
		}
		p = append([]clockPair(nil), p[start:]...)
	}
	return p
}

func (a *Alignment) trimWindow(m *clockModel) {
	m.pairs = trimPairs(m.pairs, int64(a.cfg.AlignWindow), true)
	// 斜率历史用更长跨度（时间窗的 4 倍），让毫秒级 lag 噪声在长基线上
	// 平均，从而把百分之一量级以下的缓慢漂移斜率可靠估计出来。
	histSpan := int64(a.cfg.AlignWindow) * 4
	m.hist = trimPairs(m.hist, histSpan, true)
}

func medianF(x []float64) float64 {
	y := append([]float64(nil), x...)
	sort.Float64s(y)
	return y[len(y)/2]
}

// refit 用窗内“已确认”锚点稳健重拟合并更新状态与可信判定。最新锚
// 点的统一时间可能贴近当前参考前沿、其 lag 会在未来数据补齐后被修
// 正，故拟合只采用 y 不超过 refFrontier−confirmMargin 的锚点，避免
// 暂态边缘外点污染斜率；映射换算仍可对更新时刻做短距外推。
// skewsAtBoundary 判断多数正常锚点斜率是否持续贴近可信域上界或下界
// （在一个粗搜索步长内），这是真实斜率越界、搜索被钳在边界的特征。
func (a *Alignment) skewsAtBoundary(ps []clockPair) bool {
	up := 1 + a.cfg.AlignMaxSkew
	down := 1 - a.cfg.AlignMaxSkew
	tol := 0.015
	var n, hit int
	for _, p := range ps {
		if p.diag || p.sk <= 0 {
			continue
		}
		n++
		if math.Abs(p.sk-up) <= tol || math.Abs(p.sk-down) <= tol {
			hit++
		}
	}
	return n >= 5 && hit*2 > n
}

// diagOutOfRange 判断诊断锚点是否一致（多数且至少 diagMin 个）指向
// 越界斜率。诊断锚点仅在扩展斜率范围才能对齐，故一致越界说明真实时
// 钟超出可信范围，而非样本不足。
func (a *Alignment) diagOutOfRange(ps []clockPair) bool {
	const diagMin = 5
	if len(a.refRing) == 0 {
		return false
	}
	limit := a.refRing[len(a.refRing)-1].tNS - int64(300*time.Millisecond)
	var n, bad int
	for _, p := range ps {
		if !p.diag || p.y > limit {
			continue
		}
		n++
		if math.Abs(p.sk-1) > a.cfg.AlignMaxSkew {
			bad++
		}
	}
	return n >= diagMin && bad*2 > n
}

// confirmedPairs 仅保留统一时间已被参考充分覆盖确认的锚点：诊断锚点
// 不参与正常拟合。
func (a *Alignment) confirmedPairs(m *clockModel, src []clockPair) []clockPair {
	if !a.refSet || len(a.refRing) == 0 {
		return src
	}
	limit := a.refRing[len(a.refRing)-1].tNS - int64(300*time.Millisecond)
	out := make([]clockPair, 0, len(src))
	for _, p := range src {
		if p.diag {
			continue // 诊断锚点只用于越界判定，不进正常仿射拟合
		}
		if p.y <= limit {
			out = append(out, p)
		}
	}
	return out
}

// fitOffset 在给定斜率下，取锚点残差 y−skew·x 的中位数为截距，
// 并返回残差 RMS（ns）。
func fitOffset(ps []clockPair, skew float64) (int64, float64) {
	rs := make([]float64, len(ps))
	var ss float64
	for i, p := range ps {
		rs[i] = float64(p.y) - skew*float64(p.x)
	}
	off := medianF(rs)
	for _, p := range ps {
		e := float64(p.y) - (skew*float64(p.x) + off)
		ss += e * e
	}
	return int64(math.Round(off)), math.Sqrt(ss / float64(len(ps)))
}

// lagConsensus 判断近期确认锚点的 lag(y−x) 是否“同号一致”：取 lag
// 中位数的符号为候选方向，要求至少 2/3 的锚点与该方向同号，且同号锚点
// 的 lag 中位数幅值超过 detect。零均值相关噪声符号近半翻转、过不了该
// 一致性；真实时钟跑偏的 lag 持续同号，轻松通过。
func lagConsensus(ps []clockPair, detect int64) bool {
	lag, ok := consensusLag(ps)
	return ok && math.Abs(lag) > float64(detect)
}

// consensusLag 返回近期确认锚点“同号一致”的实测时钟偏移中位数
// （单位 ns，即 lag=y−x，直接就是 unified−raw 当前偏移量，与拟合斜率
// 无关）。取 lag 中位数符号为候选方向，要求至少 2/3 锚点同号；否则
// 返回 ok=false（偏移方向不一致，判为噪声而非漂移）。
func consensusLag(ps []clockPair) (float64, bool) {
	if len(ps) < 3 {
		return 0, false
	}
	lag := make([]float64, len(ps))
	for i, p := range ps {
		lag[i] = float64(p.y - p.x)
	}
	med := medianF(lag)
	if med == 0 {
		return 0, false
	}
	dir := 1.0
	if med < 0 {
		dir = -1.0
	}
	same := make([]float64, 0, len(ps))
	for _, l := range lag {
		if dir*l > 0 {
			same = append(same, l)
		}
	}
	if len(same)*3 < len(ps)*2 {
		return 0, false
	}
	return medianF(same), true
}

func (a *Alignment) refit(m *clockModel) {
	if len(m.pairs) == 0 && len(m.hist) == 0 {
		return
	}
	confirmed := a.confirmedPairs(m, m.pairs)
	histConf := a.confirmedPairs(m, m.hist)
	minN := a.cfg.AlignMinSamples

	// 正常锚点不足。cold 态必须保守：在拿到足够多“一致且显著”的漂移
	// 证据前保持恒等（零漂移输入下相关 lag 只是 ±20ms 噪声，绝不能据此
	// 进入校准态暂缓数据，否则会改变既有融合行为）。只有诊断锚点一致
	// 指向越界斜率时才判 out_of_range。
	if len(confirmed) < minN {
		if a.diagOutOfRange(m.hist) {
			m.state = AlignOutOfRange
			return
		}
		if m.state == AlignCold {
			// 证据不足：维持恒等，不暂缓、不改映射。
			return
		}
		// calibrating 态保持校准，等待更多锚点。
		return
	}
	// 斜率直接取各锚点信号斜率比（sk）的中位数——它不经过 lag 量化，
	// 因而不受固定偏移/lag 抖动污染，能稳健反映时钟快慢。历史点更多
	// 用历史，否则用近期。
	skewSrc := histConf
	if len(skewSrc) < minN {
		skewSrc = confirmed
	}
	sks := make([]float64, 0, len(skewSrc))
	var span int64
	if len(skewSrc) > 0 {
		span = skewSrc[len(skewSrc)-1].x - skewSrc[0].x
		for _, p := range skewSrc {
			if p.sk > 0 {
				sks = append(sks, p.sk)
			}
		}
	}
	skew := m.skew
	if skew == 0 {
		skew = 1
	}
	if len(sks) >= minN {
		skew = medianF(sks)
	}
	// 偏移与残差用近期确认锚点（对当前时延最具代表性）。
	off, rms := fitOffset(confirmed, skew)
	// 越界边界检测：若足够多正常锚点的斜率都顶在可信边界附近（真实斜
	// 率在域外、受限搜索被钳在边界），判 out_of_range。这能在无需精确
	// 估计域外斜率的情况下可靠识别超界时钟。
	if a.skewsAtBoundary(confirmed) {
		m.state = AlignOutOfRange
		m.skew = skew
		m.offsetNS = off
		m.residNS = rms
		return
	}
	// 漂移判定直接用“锚点实测当前偏移”consensusLag（lag=y−x 的同号
	// 中位数），而不是仿射截距 off：在远离原点（x 很大）时斜率与截距强
	// 相关，拟合会把真实偏移吸收进斜率、使 off≈0（即便 lag 已达 0.8s），
	// 用 off 判漂移必然漏检。lag 直接就是 unified−raw 的当前偏移量，且
	// 要求同号一致（零漂移的零均值相关噪声符号近半翻转，过不了）。这保证
	// 零漂移永不离开恒等（与关闭对齐逐位一致），真实固定偏置/斜率跑偏都
	// 会迅速锁定；已锁定路若 lag 回落、一致性消失，同样退回恒等。
	detect := int64(a.cfg.AlignDetect)
	cl, consensus := consensusLag(confirmed)
	driftSeen := consensus && math.Abs(cl) > float64(detect)
	curLag := int64(math.Round(cl))
	good := span >= int64(a.cfg.AlignMinSpan) &&
		rms <= float64(a.cfg.AlignTolerance) &&
		math.Abs(skew-1) <= a.cfg.AlignMaxSkew &&
		math.Abs(float64(curLag)) <= float64(int64(a.cfg.AlignMaxDrift))
	if good && driftSeen {
		m.state = AlignLocked
		m.skew = skew
		m.offsetNS = off
		m.residNS = rms
		switch {
		case curLag > int64(lagFineStep):
			m.sign = 1
		case curLag < -int64(lagFineStep):
			m.sign = -1
		default:
			m.sign = 0
		}
		return
	}
	outOfRange := math.Abs(skew-1) > a.cfg.AlignMaxSkew ||
		(consensus && math.Abs(cl) > float64(int64(a.cfg.AlignMaxDrift)))
	switch {
	case outOfRange:
		m.state = AlignOutOfRange
	case driftSeen:
		// 已发现真实漂移但尚未拟合达标：暂缓，样本不足原因可区分。
		m.state = AlignCalibrating
	default:
		// 残差/跨度暂不达标但无显著漂移证据：保持恒等（兼容）。
		m.state = AlignCold
	}
	m.skew = skew
	m.offsetNS = off
	m.residNS = rms
}

// Stats 返回各路对齐状态快照（含参考路）。
func (a *Alignment) Stats() []AlignStat {
	out := make([]AlignStat, 0, len(a.models)+1)
	if a.refSet {
		out = append(out, AlignStat{
			Kind: a.ref, State: AlignLocked, Skew: 1, Samples: len(a.refRing), IsRef: true,
		})
	}
	for k := SensorIMU; k <= SensorAbsolutePose; k++ {
		m, ok := a.models[k]
		if !ok || (a.refSet && k == a.ref) {
			continue
		}
		out = append(out, AlignStat{
			Kind: k, State: m.state, Skew: m.skew,
			Offset: time.Duration(m.offsetNS), Residual: time.Duration(m.residNS),
			Samples: m.count,
		})
	}
	return out
}

// frontierRaw 返回某路最近（最大）原始时间戳。
func (a *Alignment) frontierRaw(k SensorKind) (time.Time, bool) {
	t, ok := a.frontier[k]
	return t, ok
}

// watermark 返回可安全消费的统一时间水线（各路对齐覆盖交集）。
//
// 口径必须与“实际消费时的换算”一致：
//   - locked：用统一换算后的该路前沿（mapRawNS）；
//   - cold（未发现漂移，恒等）：用原始前沿；
//   - calibrating（已发现漂移、样本不足）：这一路的数据正在暂缓，为不
//     让参考路（恒等、始终就绪）独自冲到前面、将来放出的旧观测全部陈旧，
//     用“原始前沿 − 搜索半径”保守约束参考消费，使时间轴在该路对齐期间
//     短暂等待而不是被带偏；该路缺测超过 DegradeAfter+SearchRange 后
//     解除约束，避免单路缺测冻结整条时间轴；
//   - out_of_range：观测持续拒绝，不参与交集（超时后），避免卡死。
func (a *Alignment) watermark() (time.Time, bool) {
	if !a.cfg.EnableAlignment || !a.refSet || len(a.refRing) == 0 {
		return time.Time{}, false
	}
	margin := int64(a.cfg.AlignReleaseMargin)
	refFront := a.refRing[len(a.refRing)-1].tNS
	wm := refFront + margin
	active := false
	constrained := false // 是否存在真正需要约束消费的（非恒等）路
	for k, m := range a.models {
		raw, ok := a.frontier[k]
		if !ok {
			continue
		}
		active = true
		// cold（未发现漂移）恒等路与参考轴同口径，不需水线；锁定路与
		// 校准/越界路都约束参考放水——保证参考 IMU 不致跑到非参考路“已确认
		// 覆盖到的统一时间”之前（否则启动/校准段的速度、位姿更新会因姿态
		// 尚未对齐而错配）。
		if m.state == AlignCold {
			continue
		}
		constrained = true
		rawNS := a.relNS(raw)
		// 某路超过“缺测宽限”(DegradeAfter+SearchRange) 未更新前沿时，
		// 视为暂时缺测：水线不再被它冻结，避免一路长时间无数据挡住其他
		// 路与参考的正常消费、撑爆待处理缓冲。
		missingFor := refFront - rawNS
		missingLimit := int64(a.cfg.DegradeAfter) + int64(a.cfg.AlignSearchRange)
		if missingFor > missingLimit {
			continue
		}
		var lim int64
		switch m.state {
		case AlignLocked:
			// 放水上界 = 最新已采纳锚点的统一时间 + 半窗。锚点之间的插值
			// 不可变、绝不回跳；最新锚点之后只允许半窗（相关居中窗宽度）的
			// 短距右外推——其误差有界（真实斜率 ±0.5% × 2.5s ≈ 12ms 量级），
			// 且下个锚点很快把它夹住。再远的点在线先暂缓（由后续批次或
			// AdvanceTo 排空），避免用会随新锚点改变的段斜率长距外推，在漂移
			// 换向点错配高频速度更新而正反馈发散。
			if _, hy, cov := a.anchorCoverage(m); cov {
				lim = hy + int64(a.halfWinFor(k)) + margin
			} else {
				lim = rawNS - int64(a.cfg.AlignSearchRange) + margin
			}
		default:
			// calibrating / out_of_range：该路观测暂缓或拒绝。用保守原始
			// 前沿约束参考（不按未校准模型外推），防止参考跑过头。
			lim = rawNS - int64(a.cfg.AlignSearchRange) + margin
		}
		if lim < wm {
			wm = lim
		}
	}
	if !active || !constrained {
		// 没有非恒等路：零漂移/全 cold，行为与关闭对齐一致（全序即时消费）。
		wm = refFront + margin
	}
	return a.fromNS(wm), true
}

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

// pendingItem 重排缓冲中的一条待消费观测。
type pendingItem struct {
	obs           Observation
	aligned       time.Time
	raw           time.Time
	ready         bool
	kind          SensorKind
	alignFallback bool // 校准宽限后未锁定，按原始时间恒等降级放行
}

// alignCheckpoint 对齐层深拷贝检查点（批预算失败时整批回滚）。
type alignCheckpoint struct {
	ref, pref SensorKind
	refSet    bool
	hasPref   bool
	autoRef   bool
	origin    time.Time
	originSet bool
	refRing   []sigSample
	models    map[SensorKind]clockModel
	frontier  map[SensorKind]time.Time
}

// Checkpoint / Restore 捕获与恢复对齐状态。
func (a *Alignment) Checkpoint() alignCheckpoint {
	cp := alignCheckpoint{
		ref: a.ref, pref: a.preferred, refSet: a.refSet, hasPref: a.hasPref, autoRef: a.autoRef,
		origin:    a.origin,
		originSet: a.originSet,
		refRing:   append([]sigSample(nil), a.refRing...),
		models:    make(map[SensorKind]clockModel, len(a.models)),
		frontier:  make(map[SensorKind]time.Time, len(a.frontier)),
	}
	for k, m := range a.models {
		cp.models[k] = clockModel{
			state: m.state, skew: m.skew, offsetNS: m.offsetNS, residNS: m.residNS,
			pairs: append([]clockPair(nil), m.pairs...), hist: append([]clockPair(nil), m.hist...), count: m.count,
			sign: m.sign, lastAnch: m.lastAnch, hasLastAnch: m.hasLastAnch, anchoredAt: m.anchoredAt, lastAdopt: m.lastAdopt,
			firstSeen: m.firstSeen, hasFirst: m.hasFirst,
			sig: append([]sigSample(nil), m.sig...),
		}
	}
	for k, v := range a.frontier {
		cp.frontier[k] = v
	}
	return cp
}

func (a *Alignment) Restore(cp alignCheckpoint) {
	// 参考路是统一时间基准：一旦建立只能向前推进，单批预算回滚绝不
	// 撤销已建立的参考选择、原点与信号样本，否则后续所有配对都会被
	// 错位的时间基准污染。只回滚非参考路的时钟模型与各路前沿。
	if !a.refSet && cp.refSet {
		a.ref, a.preferred = cp.ref, cp.pref
		a.refSet, a.hasPref, a.autoRef = cp.refSet, cp.hasPref, cp.autoRef
	}
	if !a.originSet && cp.originSet {
		a.origin, a.originSet = cp.origin, cp.originSet
	}
	a.models = make(map[SensorKind]*clockModel, len(cp.models))
	for k, m := range cp.models {
		if a.refSet && k == a.ref {
			continue
		}
		mm := m
		mm.pairs = append([]clockPair(nil), m.pairs...)
		mm.hist = append([]clockPair(nil), m.hist...)
		mm.sig = append([]sigSample(nil), m.sig...)
		a.models[k] = &mm
	}
	a.frontier = make(map[SensorKind]time.Time, len(cp.frontier))
	for k, v := range cp.frontier {
		if a.refSet && k == a.ref {
			continue
		}
		a.frontier[k] = v
	}
}
