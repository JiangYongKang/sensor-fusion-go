// Package timesync estimates a per-stream affine clock model
//
//	localNS = (1 + drift) * nativeNS + offsetNS
//
// from paired (native, reference) samples using a deterministic two-point /
// least-squares estimator. It provides reproducible correction of constant
// clock offset and first-order clock drift.
package timesync

import "math"

// Model is an affine clock model for one stream.
type Model struct {
	OffsetNS float64
	Drift    float64 // dimensionless, e.g. 1e-5
	Ready    bool
	Samples  int
}

// Estimator accumulates paired time samples and fits an affine model.
type Estimator struct {
	streamID int
	minPairs int
	maxPairs int
	maxDrift float64
	pairs    [][2]int64 // (nativeNS, referenceNS)
}

// Config configures a clock estimator.
type Config struct {
	MinPairs int     // pairs required before Ready
	MaxPairs int     // ring capacity (oldest replaced)
	MaxDrift float64 // hard sanity bound on |drift|
}

// NewEstimator builds an estimator. Zero/negative values receive safe
// defaults so behavior is always defined.
func NewEstimator(streamID int, cfg Config) *Estimator {
	if cfg.MinPairs <= 0 {
		cfg.MinPairs = 2
	}
	if cfg.MaxPairs < cfg.MinPairs {
		cfg.MaxPairs = cfg.MinPairs
	}
	if cfg.MaxDrift <= 0 {
		cfg.MaxDrift = 1e-3
	}
	return &Estimator{
		streamID: streamID,
		minPairs: cfg.MinPairs,
		maxPairs: cfg.MaxPairs,
		maxDrift: cfg.MaxDrift,
	}
}

// AddSample records a (native, reference) pair. Only pairs whose native
// timestamp is strictly newer than the last accepted one are used: clock
// fitting is a relationship between two clocks and must not be fed
// out-of-order deliveries (which would produce meaningless, huge slopes).
// When the ring is full the oldest pair is replaced (FIFO). Returns true
// when the fitted model changed state.
func (e *Estimator) AddSample(nativeNS, referenceNS int64) bool {
	before := e.fit()
	if n := len(e.pairs); n > 0 && nativeNS <= e.pairs[n-1][0] {
		return false
	}
	if len(e.pairs) >= e.maxPairs {
		e.pairs = e.pairs[1:]
	}
	e.pairs = append(e.pairs, [2]int64{nativeNS, referenceNS})
	after := e.fit()
	if before.Ready != after.Ready {
		return true
	}
	return math.Abs(before.OffsetNS-after.OffsetNS) > 1 || math.Abs(before.Drift-after.Drift) > 1e-12
}

// fit performs an ordinary least-squares fit reference = a*native + b in
// float64. To keep conditioning identical across runs the computation is
// done in native units (nanoseconds) with a centering step.
func (e *Estimator) fit() Model {
	n := len(e.pairs)
	m := Model{Samples: n}
	if n < e.minPairs {
		return m
	}
	// Center on the first sample to avoid catastrophic cancellation.
	n0 := float64(e.pairs[0][0])
	var sx, sy float64
	xs := make([]float64, n)
	for i, p := range e.pairs {
		xs[i] = float64(p[0]) - n0
		sy += float64(p[1])
		sx += xs[i]
	}
	mx, my := sx/float64(n), sy/float64(n)
	var sxx, sxy float64
	for i, p := range e.pairs {
		dx := xs[i] - mx
		sxx += dx * dx
		sxy += dx * (float64(p[1]) - my)
	}
	a := 1.0
	if sxx > 0 {
		a = sxy / sxx
	}
	if math.IsNaN(a) || math.IsInf(a, 0) || math.Abs(a-1) > e.maxDrift {
		// Implausible fit: refuse to mark ready rather than warp time.
		return Model{Samples: n}
	}
	b := my - a*(mx+n0)
	m.OffsetNS = b
	m.Drift = a - 1
	m.Ready = true
	return m
}

// Model returns the current fitted model.
func (e *Estimator) Model() Model { return e.fit() }

// ToLocal maps a native timestamp onto the reference clock. Rounding is
// round-to-nearest-even-free (math.Round) and therefore deterministic.
func (m Model) ToLocal(nativeNS int64) (int64, bool) {
	if !m.Ready {
		return 0, false
	}
	y := (1+m.Drift)*float64(nativeNS) + m.OffsetNS
	if math.IsNaN(y) || math.IsInf(y, 0) {
		return 0, false
	}
	return int64(math.Round(y)), true
}
