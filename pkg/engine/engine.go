// Package engine wires time synchronisation, reorder buffering, gating,
// state estimation, sensor degradation, bounded resources and concurrency
// control into one local state-estimation engine.
//
// All state transitions happen under a single mutex: concurrent Predict /
// Update / Snapshot calls can never observe a half-updated state, and the
// same input sequence always produces the same output sequence regardless
// of goroutine scheduling. Timing-budget enforcement uses a deterministic
// injected clock function (the wall clock by default) and the processing
// order itself never depends on timing: when the budget is exhausted, only
// fully-complete processing boundaries are returned to the caller and the
// not-yet-started observation stays buffered for the next AdvanceTo.
package engine

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/highcumontoa/sensor-fusion-go/pkg/buffer"
	"github.com/highcumontoa/sensor-fusion-go/pkg/fusion"
	"github.com/highcumontoa/sensor-fusion-go/pkg/outlier"
	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
	"github.com/highcumontoa/sensor-fusion-go/pkg/timesync"
)

// StreamMode is the degradation state of a stream.
type StreamMode int

const (
	StreamOK StreamMode = iota
	StreamUnavailable
	StreamDegraded
	StreamStale
)

func (m StreamMode) String() string {
	switch m {
	case StreamOK:
		return "ok"
	case StreamUnavailable:
		return "unavailable"
	case StreamDegraded:
		return "degraded"
	case StreamStale:
		return "stale"
	default:
		return "unknown"
	}
}

// StreamHealth describes the current usability of one stream.
type StreamHealth struct {
	Stream   sensors.Stream
	Mode     StreamMode
	LastNS   int64 // reference time of last successfully consumed sample
	AgeNS    int64 // now - LastNS
	Samples  int64
	Degraded bool
}

// Limits bounds input scale and per-step processing cost.
type Limits struct {
	MaxPending      int           // reject inserts beyond this many buffered items
	MaxObservations int64         // reject observations after this many accepted
	StepBudget      time.Duration // max wall time spent processing one Submit/Advance
}

// DegradeRule configures aging behaviour of one stream.
type DegradeRule struct {
	DegradeAgeNS int64   // age beyond which measurement sigma is inflated
	StaleAgeNS   int64   // age beyond which the stream is considered unusable
	SigmaFactor  float64 // uncertainty inflation factor while degraded
}

// Config is the full engine configuration.
type Config struct {
	StartNS int64
	Buffer  struct {
		WindowNS int64
		Capacity int
	}
	Clock map[sensors.Stream]timesync.Config
	// LinkDelayNS is the known fixed measurement->arrival latency per
	// stream. It is subtracted from ArrivalRefNS before fitting the clock
	// model, so a constant transport delay is never mistaken for a clock
	// offset (which would couple into drift and bias long runs).
	LinkDelayNS map[sensors.Stream]int64
	Fusion      fusion.Config
	Degrade     map[sensors.Stream]DegradeRule
	Limits      Limits
	// MaxAudit caps retained decision records (oldest dropped); <=0 = 4096.
	MaxAudit int
	// ClockFn returns wall time for budget enforcement; tests inject a
	// deterministic monotonic clock.
	ClockFn func() time.Time
}

// Engine is the concurrency-safe state estimation engine.
type Engine struct {
	mu       sync.Mutex
	cfg      Config
	buf      *buffer.Buffer
	clocks   map[sensors.Stream]*timesync.Estimator
	filter   *fusion.Filter
	health   map[sensors.Stream]*streamState
	accepted int64
	rejected int64
	lastIMU  *sensors.Observation
	lastNow  int64
	audit    []sensors.Decision
	closed   bool
	clockFn  func() time.Time
}

type streamState struct {
	lastNS   int64
	samples  int64
	mode     StreamMode
	lastSeq  int64
	haveSeq  bool
	gapCount int64
}

// New builds an engine with safe defaults for any unset rules.
func New(cfg Config) *Engine {
	if cfg.Buffer.Capacity == 0 {
		cfg.Buffer.Capacity = 8192
	}
	if cfg.MaxAudit <= 0 {
		cfg.MaxAudit = 4096
	}
	if cfg.Clock == nil {
		cfg.Clock = map[sensors.Stream]timesync.Config{}
	}
	if cfg.Degrade == nil {
		cfg.Degrade = map[sensors.Stream]DegradeRule{}
	}
	if cfg.LinkDelayNS == nil {
		cfg.LinkDelayNS = map[sensors.Stream]int64{}
	}
	if cfg.ClockFn == nil {
		cfg.ClockFn = time.Now
	}
	e := &Engine{
		cfg:     cfg,
		buf:     buffer.New(buffer.Config{WindowNS: cfg.Buffer.WindowNS, Capacity: cfg.Buffer.Capacity}),
		clocks:  make(map[sensors.Stream]*timesync.Estimator),
		filter:  fusion.New(cfg.Fusion, cfg.StartNS),
		health:  make(map[sensors.Stream]*streamState),
		lastNow: cfg.StartNS,
		clockFn: cfg.ClockFn,
	}
	for _, s := range sensors.AllStreams {
		cc, ok := cfg.Clock[s]
		if !ok {
			cc = timesync.Config{MinPairs: 2, MaxPairs: 32, MaxDrift: 1e-3}
		}
		e.clocks[s] = timesync.NewEstimator(int(s), cc)
		e.health[s] = &streamState{lastNS: cfg.StartNS, mode: StreamUnavailable}
	}
	return e
}

// Report is the outcome of one Submit.
type Report struct {
	Decision sensors.Decision
	Released []sensors.Decision // decisions for observations released & processed
}

func (e *Engine) record(d sensors.Decision) {
	e.audit = append(e.audit, d)
	if len(e.audit) > e.cfg.MaxAudit {
		e.audit = e.audit[len(e.audit)-e.cfg.MaxAudit:]
	}
}

// reject constructs a rejection decision and counts it.
func (e *Engine) reject(o sensors.Observation, localNS int64, r sensors.Reason, detail string) Report {
	d := sensors.Decision{ID: o.ID, Stream: o.Stream, LocalNS: localNS, Accepted: false, Reason: r, Detail: detail}
	e.rejected++
	e.record(d)
	return Report{Decision: d}
}

// Submit ingests one observation. It is first clock-corrected and buffered;
// any observations whose reorder window has elapsed are immediately released
// and processed in strict time order. Submit is serialised and atomic with
// respect to Snapshot.
func (e *Engine) Submit(ctx context.Context, o sensors.Observation) Report {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return e.reject(o, o.LocalNS, sensors.ReasonUnsupported, "engine closed")
	}
	if err := ctx.Err(); err != nil {
		return e.reject(o, o.LocalNS, sensors.ReasonProcessingBudget, err.Error())
	}
	// 1. Clock correction.
	est := e.clocks[o.Stream]
	if est == nil {
		return e.reject(o, o.LocalNS, sensors.ReasonUnsupported, "unknown stream")
	}
	if o.ArrivalRefNS != 0 {
		refAtMeasure := o.ArrivalRefNS - e.cfg.LinkDelayNS[o.Stream]
		est.AddSample(o.NativeNS, refAtMeasure)
	}
	localNS, ok := est.Model().ToLocal(o.NativeNS)
	if !ok {
		if o.LocalNS != 0 {
			// Pre-synchronised fallback.
			localNS = o.LocalNS
		} else {
			return e.reject(o, 0, sensors.ReasonClockNotReady,
				fmt.Sprintf("clock model for %s has %d samples", o.Stream, est.Model().Samples))
		}
	}
	o.LocalNS = localNS
	if localNS < e.cfg.StartNS {
		return e.reject(o, localNS, sensors.ReasonBeforeStart,
			fmt.Sprintf("t=%d before start=%d", localNS, e.cfg.StartNS))
	}
	// Size limits count every submission attempt (accepted or rejected), so
	// an over-limit caller can never grow engine state through retries.
	if e.cfg.Limits.MaxPending > 0 && e.buf.Len() >= e.cfg.Limits.MaxPending {
		return e.reject(o, localNS, sensors.ReasonBufferBackpressure,
			fmt.Sprintf("pending limit %d reached", e.cfg.Limits.MaxPending))
	}
	if e.cfg.Limits.MaxObservations > 0 && e.accepted+e.rejected >= e.cfg.Limits.MaxObservations {
		return e.reject(o, localNS, sensors.ReasonSizeLimit,
			fmt.Sprintf("total observations limit %d reached", e.cfg.Limits.MaxObservations))
	}
	// 2. Ordered insert.
	out := e.buf.Insert(o)
	var primary sensors.Decision
	switch out {
	case buffer.InsertDuplicate:
		return e.reject(o, localNS, sensors.ReasonDuplicate,
			fmt.Sprintf("duplicate (stream=%s,nativeNS=%d)", o.Stream, o.NativeNS))
	case buffer.InsertLate:
		return e.reject(o, localNS, sensors.ReasonDroppedLate,
			fmt.Sprintf("t=%d behind released watermark=%d", localNS, e.buf.Watermark()))
	case buffer.InsertBackpressure:
		return e.reject(o, localNS, sensors.ReasonBufferBackpressure,
			fmt.Sprintf("buffer capacity %d exhausted", e.cfg.Buffer.Capacity))
	case buffer.InsertAcceptedLate:
		primary = sensors.Decision{ID: o.ID, Stream: o.Stream, LocalNS: localNS,
			Accepted: true, Reason: sensors.ReasonOutOfOrder,
			Detail: fmt.Sprintf("late arrival t=%d inserted ahead of watermark=%d", localNS, e.buf.Watermark())}
	default:
		primary = sensors.Decision{ID: o.ID, Stream: o.Stream, LocalNS: localNS,
			Accepted: true, Reason: sensors.ReasonOK, Detail: "buffered in order"}
	}
	e.record(primary)

	// 3. Release everything the reorder window now permits.
	deadline := e.deadline()
	released := e.processReady(o.ArrivalRefNS, deadline)
	if e.budgetExceeded(deadline) && e.buf.Len() > 0 {
		e.record(sensors.Decision{Stream: o.Stream, LocalNS: localNS,
			Reason: sensors.ReasonProcessingBudget,
			Detail: fmt.Sprintf("step budget %s exhausted; %d items retained in order",
				e.cfg.Limits.StepBudget, e.buf.Len())})
	}
	return Report{Decision: primary, Released: released}
}

func (e *Engine) deadline() time.Time {
	if e.cfg.Limits.StepBudget > 0 {
		return e.clockFn().Add(e.cfg.Limits.StepBudget)
	}
	return time.Time{}
}

func (e *Engine) budgetExceeded(deadline time.Time) bool {
	return !deadline.IsZero() && e.clockFn().After(deadline)
}

// processReady consumes buffered observations in time order while the
// budget holds. Every consumed item is either fully applied (state +
// covariance + health + accounting together) or fully recorded as rejected;
// there is no half-applied boundary.
func (e *Engine) processReady(nowNS int64, deadline time.Time) []sensors.Decision {
	var decisions []sensors.Decision
	for {
		if !deadline.IsZero() && e.clockFn().After(deadline) {
			return decisions
		}
		if nowNS < e.lastNow {
			nowNS = e.lastNow
		}
		o, ok := e.buf.PopReady(nowNS)
		if !ok {
			return decisions
		}
		d := e.consume(o)
		decisions = append(decisions, d)
		e.record(d)
	}
}

func (e *Engine) consume(o sensors.Observation) sensors.Decision {
	st := e.health[o.Stream]
	// Sequence-gap accounting (deterministic, informational only).
	if st.haveSeq && o.Seq > st.lastSeq+1 {
		st.gapCount += o.Seq - st.lastSeq - 1
	}
	st.haveSeq = true
	st.lastSeq = o.Seq

	// Validity / payload checks.
	if reason, detail := payloadReason(o); reason != sensors.ReasonOK {
		e.rejected++
		st.mode = StreamStale
		return sensors.Decision{ID: o.ID, Stream: o.Stream, LocalNS: o.LocalNS,
			Accepted: false, Reason: reason, Detail: detail}
	}

	// Predict state forward in time (all observations drive the clock).
	if !e.filter.PredictTo(o.LocalNS, e.lastIMU) {
		e.rejected++
		return sensors.Decision{ID: o.ID, Stream: o.Stream, LocalNS: o.LocalNS,
			Accepted: false, Reason: sensors.ReasonTimeRegress,
			Detail: fmt.Sprintf("filter time regression to t=%d", o.LocalNS)}
	}
	e.lastNow = o.LocalNS

	// Aging/degradation for this stream.
	rule := e.cfg.Degrade[o.Stream]
	age := o.LocalNS - st.lastNS
	mode := StreamOK
	detail := ""
	if st.samples == 0 {
		mode = StreamOK
	} else if rule.StaleAgeNS > 0 && age > rule.StaleAgeNS {
		// The sample itself is fresh relative to now, but it resumes after a
		// stale gap: accept it but flag the stream transition explicitly.
		mode = StreamStale
		detail = fmt.Sprintf("resumed after stale gap age=%dms", age/1_000_000)
	} else if rule.DegradeAgeNS > 0 && age > rule.DegradeAgeNS {
		mode = StreamDegraded
		detail = fmt.Sprintf("gap age=%dms exceeds degrade threshold; sigma x%.2f",
			age/1_000_000, rule.SigmaFactor)
		o = inflateSigma(o, rule.SigmaFactor)
	}
	st.mode = mode

	var d sensors.Decision
	switch o.Stream {
	case sensors.StreamIMU:
		imu := o
		e.lastIMU = &imu
		st.samples++
		st.lastNS = o.LocalNS
		reason := sensors.ReasonOK
		if mode == StreamStale {
			reason = sensors.ReasonStale
		} else if mode == StreamDegraded {
			reason = sensors.ReasonDegraded
		}
		d = sensors.Decision{ID: o.ID, Stream: o.Stream, LocalNS: o.LocalNS,
			Accepted: true, Reason: reason, Detail: withPrefix(detail, "IMU propagated")}
		e.accepted++
	case sensors.StreamOdom:
		res := e.filter.UpdateVelocity(o)
		d = e.finalizeUpdate(o, res, mode, detail)
	case sensors.StreamAbsPose:
		res := e.filter.UpdateAbs(o)
		d = e.finalizeUpdate(o, res, mode, detail)
	}
	if d.Accepted {
		st.samples++
		st.lastNS = o.LocalNS
		st.mode = mode
	}
	return d
}

func withPrefix(prefix, msg string) string {
	if prefix == "" {
		return msg
	}
	return prefix + "; " + msg
}

func (e *Engine) finalizeUpdate(o sensors.Observation, res fusion.UpdateResult, mode StreamMode, modeDetail string) sensors.Decision {
	d := sensors.Decision{ID: o.ID, Stream: o.Stream, LocalNS: o.LocalNS}
	if !res.Applied {
		e.rejected++
		d.Accepted = false
		d.Reason = sensors.ReasonGateReject
		if res.Verdict == outlier.VerdictRejectDegenerate {
			d.Detail = "degenerate innovation covariance; " + res.Detail
		} else {
			d.Detail = res.Detail
		}
		return d
	}
	e.accepted++
	d.Accepted = true
	switch mode {
	case StreamStale:
		d.Reason = sensors.ReasonStale
	case StreamDegraded:
		d.Reason = sensors.ReasonDegraded
	default:
		d.Reason = sensors.ReasonOK
	}
	d.Detail = withPrefix(modeDetail, res.Detail)
	return d
}

func inflateSigma(o sensors.Observation, f float64) sensors.Observation {
	if f <= 1 {
		return o
	}
	o.SigmaPos *= f
	o.SigmaAng *= f
	o.SigmaV *= f
	return o
}

func payloadReason(o sensors.Observation) (sensors.Reason, string) {
	switch o.Stream {
	case sensors.StreamIMU:
		if o.SigmaIMUA <= 0 || o.SigmaIMUG <= 0 {
			return sensors.ReasonUnsupported, "IMU record missing noise parameters"
		}
	case sensors.StreamOdom:
		if o.SigmaV <= 0 {
			return sensors.ReasonUnsupported, "odometry record missing velocity sigma"
		}
	case sensors.StreamAbsPose:
		if !o.HasAbsPose {
			return sensors.ReasonUnsupported, "absolute-pose record has no pose payload"
		}
		if o.SigmaPos <= 0 || o.SigmaAng <= 0 {
			return sensors.ReasonUnsupported, "absolute-pose record missing sigmas"
		}
		if o.Quat[0] == 0 && o.Quat[1] == 0 && o.Quat[2] == 0 && o.Quat[3] == 0 {
			return sensors.ReasonUnsupported, "absolute-pose quaternion is zero"
		}
	default:
		return sensors.ReasonUnsupported, "unknown stream"
	}
	return sensors.ReasonOK, ""
}

// AdvanceTo drives time to nowNS and processes every buffered observation
// whose reorder window has elapsed by then.
func (e *Engine) AdvanceTo(ctx context.Context, nowNS int64) []sensors.Decision {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := ctx.Err(); err != nil {
		d := sensors.Decision{Accepted: false, Reason: sensors.ReasonProcessingBudget, Detail: err.Error()}
		e.record(d)
		return []sensors.Decision{d}
	}
	deadline := e.deadline()
	d := e.processReady(nowNS, deadline)
	e.lastNow = nowNS
	// Refresh health modes based on age even if a stream produced nothing.
	e.refreshModes(nowNS)
	return d
}

func (e *Engine) refreshModes(nowNS int64) {
	for s, st := range e.health {
		if st.samples == 0 {
			st.mode = StreamUnavailable
			continue
		}
		age := nowNS - st.lastNS
		rule := e.cfg.Degrade[s]
		if rule.StaleAgeNS > 0 && age > rule.StaleAgeNS {
			st.mode = StreamStale
		} else if rule.DegradeAgeNS > 0 && age > rule.DegradeAgeNS {
			st.mode = StreamDegraded
		} else {
			st.mode = StreamOK
		}
	}
}

// Snapshot returns a defensive copy of the filter state.
func (e *Engine) Snapshot() fusion.State {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.filter.Snapshot()
}

// Health returns a defensive copy of per-stream health.
func (e *Engine) Health() []StreamHealth {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]StreamHealth, 0, len(e.health))
	for s, st := range e.health {
		age := e.lastNow - st.lastNS
		if st.samples == 0 {
			age = -1
		}
		out = append(out, StreamHealth{
			Stream: s, Mode: st.mode, LastNS: st.lastNS, AgeNS: age,
			Samples: st.samples, Degraded: st.mode == StreamDegraded,
		})
	}
	return out
}

// Audit returns a copy of the recorded decision trail.
func (e *Engine) Audit() []sensors.Decision {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]sensors.Decision, len(e.audit))
	copy(out, e.audit)
	return out
}

// AcceptedCount reports fully consumed observations.
func (e *Engine) AcceptedCount() int64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.accepted
}

// RejectedCount reports observations that were never applied.
func (e *Engine) RejectedCount() int64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.rejected
}

// Close releases retained resources. After Close all Submits are rejected.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	e.buf.Close()
	e.audit = nil
	return nil
}
