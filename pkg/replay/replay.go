// Package replay generates deterministic synthetic sensor data from a
// scripted ground-truth trajectory. Generation is controlled by a fixed
// seed, so every replay run yields the identical observation sequence
// (including out-of-order delivery, dropped frames, outliers and clock
// drift), making results reproducible.
package replay

import (
	"math"
	"sort"

	"github.com/highcumontoa/sensor-fusion-go/pkg/pose"
	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
)

// ClockModel describes a stream clock relative to the reference clock.
type ClockModel struct {
	OffsetNS int64
	DriftPPM float64 // parts per million
}

// Config controls the deterministic replay.
type Config struct {
	Seed         int64
	DurationNS   int64
	IMUPeriodNS  int64
	OdomPeriodNS int64
	AbsPeriodNS  int64
	// Clock models per stream.
	IMUClock  ClockModel
	OdomClock ClockModel
	AbsClock  ClockModel
	// Delivery perturbation.
	OutOfOrderEvery   int   // >0: every kth delivered frame is delayed
	OutOfOrderDelayNS int64 // delay applied to those frames
	DropEveryN        int   // >0: every nth source frame is dropped
	OutlierEveryN     int   // >0: every nth delivered pose/odom is an outlier
	OutlierPosM       float64
	OutlierAngRad     float64
	LinkDelayNS       int64 // fixed measurement-to-delivery delay
	// Noise sigmas (1-sigma) baked into the generated measurements.
	SigmaAcc, SigmaGyr float64
	SigmaV             float64
	SigmaPos, SigmaAng float64
}

// TruthPoint is one ground-truth trajectory sample.
type TruthPoint struct {
	TNS  int64
	Pos  [3]float64
	Quat [4]float64
	Vel  [3]float64
	Gyr  [3]float64
}

// Generator produces the replay stream.
type Generator struct{ cfg Config }

// Result bundles a generated dataset.
type Result struct {
	Order []sensors.Observation // delivery order (may be out of order)
	Truth []TruthPoint
}

type rng struct {
	state uint64
	have  bool
	g     float64
}

func newRNG(seed int64) *rng { return &rng{state: uint64(seed) ^ 0x9E3779B97F4A7C15} }

// next returns a uniform value in [0,1) via splitmix64 + float conversion.
func (r *rng) next() float64 {
	r.state += 0x9E3779B97F4A7C15
	z := r.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	z = z ^ (z >> 31)
	return float64(z>>11) / float64(1<<53)
}

func (r *rng) gaussian() float64 {
	if r.have {
		r.have = false
		return r.g
	}
	u1, u2 := r.next(), r.next()
	if u1 < 1e-15 {
		u1 = 1e-15
	}
	r0 := math.Sqrt(-2 * math.Log(u1))
	g1 := r0 * math.Cos(2*math.Pi*u2)
	r.g = r0 * math.Sin(2*math.Pi*u2)
	r.have = true
	return g1
}

// DefaultConfig returns a representative deterministic scenario.
func DefaultConfig() Config {
	return Config{
		Seed:              20260921,
		DurationNS:        20_000_000_000, // 20 s
		IMUPeriodNS:       5_000_000,      // 200 Hz
		OdomPeriodNS:      50_000_000,     // 20 Hz
		AbsPeriodNS:       200_000_000,    // 5 Hz
		IMUClock:          ClockModel{OffsetNS: 12_000, DriftPPM: 8},
		OdomClock:         ClockModel{OffsetNS: -30_000, DriftPPM: -15},
		AbsClock:          ClockModel{OffsetNS: 5_000, DriftPPM: 2},
		OutOfOrderEvery:   17,
		OutOfOrderDelayNS: 120_000_000,
		DropEveryN:        23,
		OutlierEveryN:     11,
		OutlierPosM:       8,
		OutlierAngRad:     0.6,
		LinkDelayNS:       2_000_000,
		SigmaAcc:          0.05,
		SigmaGyr:          1e-3,
		SigmaV:            0.03,
		SigmaPos:          0.02,
		SigmaAng:          0.01,
	}
}

func New(cfg Config) *Generator {
	if cfg.IMUPeriodNS <= 0 || cfg.OdomPeriodNS <= 0 || cfg.AbsPeriodNS <= 0 {
		d := DefaultConfig()
		if cfg.IMUPeriodNS <= 0 {
			cfg.IMUPeriodNS = d.IMUPeriodNS
		}
		if cfg.OdomPeriodNS <= 0 {
			cfg.OdomPeriodNS = d.OdomPeriodNS
		}
		if cfg.AbsPeriodNS <= 0 {
			cfg.AbsPeriodNS = d.AbsPeriodNS
		}
		if cfg.DurationNS <= 0 {
			cfg.DurationNS = d.DurationNS
		}
	}
	return &Generator{cfg: cfg}
}

// truthAt evaluates the scripted ground-truth trajectory: slow straight
// motion plus a smooth combined-axis turn. Everything is analytic so truth
// is available at arbitrary times without interpolation error.
func (g *Generator) truthAt(tNS int64) TruthPoint {
	t := float64(tNS) * 1e-9
	yaw := 0.15 * t
	// Position: forward (world X) plus a slow sinusoid in Y.
	p := pose.Vec3{
		X: 0.6 * t,
		Y: 0.25 * math.Sin(0.5*t),
		Z: 0.02 * t,
	}
	v := pose.Vec3{
		X: 0.6,
		Y: 0.125 * math.Cos(0.5*t),
		Z: 0.02,
	}
	qz := pose.QuatFromAxisAngle(pose.Vec(0, 0, 1), yaw)
	qx := pose.QuatFromAxisAngle(pose.Vec(1, 0, 0), 0.05*math.Sin(0.3*t))
	q := qz.Mul(qx)
	gyr := pose.Vec(0.015*math.Cos(0.3*t), 0, 0.15)
	return TruthPoint{
		TNS:  tNS,
		Pos:  p.Slice(),
		Quat: [4]float64{q.W, q.X, q.Y, q.Z},
		Vel:  v.Slice(),
		Gyr:  gyr.Slice(),
	}
}

// TruthAt exposes ground truth at an arbitrary time.
func (g *Generator) TruthAt(tNS int64) TruthPoint { return g.truthAt(tNS) }

func nativeTime(refNS int64, c ClockModel) int64 {
	drift := c.DriftPPM * 1e-6
	// local = (1+drift)*native + offset  =>  native = (local-offset)/(1+drift)
	n := (float64(refNS) - float64(c.OffsetNS)) / (1 + drift)
	return int64(math.Round(n))
}

type delivery struct {
	o       sensors.Observation
	arrival int64
}

// Generate builds the full dataset in delivery order.
func (g *Generator) Generate() Result {
	c := g.cfg
	r := newRNG(c.Seed)
	var deliv []delivery
	var truth []TruthPoint
	var id int64

	add := func(period int64, clock ClockModel, kind sensors.Stream, build func(tr TruthPoint) sensors.Observation, seq *int64) {
		var n int64
		for t := period; t <= c.DurationNS; t += period {
			n++
			// Frame drop happens on the source stream.
			if c.DropEveryN > 0 && n%int64(c.DropEveryN) == 0 {
				continue
			}
			tr := g.truthAt(t)
			o := build(tr)
			o.ID = id
			id++
			*seq++
			o.Seq = *seq
			o.Stream = kind
			o.NativeNS = nativeTime(t, clock)
			o.ArrivalRefNS = t + c.LinkDelayNS
			if c.OutOfOrderEvery > 0 && n%int64(c.OutOfOrderEvery) == 0 {
				o.ArrivalRefNS += c.OutOfOrderDelayNS
			}
			deliv = append(deliv, delivery{o: o, arrival: o.ArrivalRefNS})
		}
	}

	var seqIMU, seqOdom, seqAbs int64
	add(c.IMUPeriodNS, c.IMUClock, sensors.StreamIMU, func(tr TruthPoint) sensors.Observation {
		return sensors.Observation{
			Acc: [3]float64{
				c.SigmaAcc * r.gaussian(),
				c.SigmaAcc * r.gaussian(),
				9.81 + c.SigmaAcc*r.gaussian(),
			},
			Gyr:       [3]float64{tr.Gyr[0] + c.SigmaGyr*r.gaussian(), c.SigmaGyr * r.gaussian(), tr.Gyr[2] + c.SigmaGyr*r.gaussian()},
			SigmaIMUA: c.SigmaAcc,
			SigmaIMUG: c.SigmaGyr,
		}
	}, &seqIMU)

	add(c.OdomPeriodNS, c.OdomClock, sensors.StreamOdom, func(tr TruthPoint) sensors.Observation {
		o := sensors.Observation{SigmaV: c.SigmaV}
		// Body-frame forward velocity; rotate truth v into body frame.
		q := pose.Quat{W: tr.Quat[0], X: tr.Quat[1], Y: tr.Quat[2], Z: tr.Quat[3]}
		vb := q.Conj().Rotate(pose.VecFrom(tr.Vel))
		o.OdomV = [3]float64{
			vb.X + c.SigmaV*r.gaussian(),
			vb.Y + c.SigmaV*r.gaussian(),
			vb.Z + c.SigmaV*r.gaussian(),
		}
		return o
	}, &seqOdom)

	add(c.AbsPeriodNS, c.AbsClock, sensors.StreamAbsPose, func(tr TruthPoint) sensors.Observation {
		o := sensors.Observation{
			HasAbsPose: true,
			SigmaPos:   c.SigmaPos,
			SigmaAng:   c.SigmaAng,
		}
		o.Pos = [3]float64{
			tr.Pos[0] + c.SigmaPos*r.gaussian(),
			tr.Pos[1] + c.SigmaPos*r.gaussian(),
			tr.Pos[2] + c.SigmaPos*r.gaussian(),
		}
		q := pose.Quat{W: tr.Quat[0], X: tr.Quat[1], Y: tr.Quat[2], Z: tr.Quat[3]}
		q = q.Mul(pose.QuatFromAxisAngle(pose.Vec(0, 0, 1), c.SigmaAng*r.gaussian()))
		o.Quat = [4]float64{q.W, q.X, q.Y, q.Z}
		return o
	}, &seqAbs)

	// Deterministic injection of outliers on delivered pose/odom frames.
	var poseLike int64
	for i := range deliv {
		if deliv[i].o.Stream != sensors.StreamAbsPose {
			continue
		}
		poseLike++
		if c.OutlierEveryN > 0 && poseLike%int64(c.OutlierEveryN) == 0 {
			o := deliv[i].o
			o.Pos[0] += c.OutlierPosM
			o.Pos[1] += 0.5 * c.OutlierPosM
			if c.OutlierAngRad != 0 {
				q := pose.Quat{W: o.Quat[0], X: o.Quat[1], Y: o.Quat[2], Z: o.Quat[3]}
				q = q.Mul(pose.QuatFromAxisAngle(pose.Vec(0, 0, 1), c.OutlierAngRad))
				o.Quat = [4]float64{q.W, q.X, q.Y, q.Z}
			}
			deliv[i].o = o
		}
	}

	// Delivery order is arrival order; tie-break by stream, id.
	sort.SliceStable(deliv, func(i, j int) bool {
		if deliv[i].arrival != deliv[j].arrival {
			return deliv[i].arrival < deliv[j].arrival
		}
		if deliv[i].o.Stream != deliv[j].o.Stream {
			return deliv[i].o.Stream < deliv[j].o.Stream
		}
		return deliv[i].o.ID < deliv[j].o.ID
	})

	res := Result{Order: make([]sensors.Observation, len(deliv))}
	// Truth samples on a 10 ms grid for error evaluation.
	for t := int64(0); t <= c.DurationNS; t += 10_000_000 {
		truth = append(truth, g.truthAt(t))
	}
	for i, d := range deliv {
		res.Order[i] = d.o
	}
	res.Truth = truth
	return res
}
