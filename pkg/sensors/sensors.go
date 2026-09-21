// Package sensors defines the unified observation record accepted by the
// fusion engine and the distinguishable decision reasons reported whenever
// an observation is accepted, rejected, deferred or causes a degradation.
package sensors

import "time"

// Stream identifies a sensor input stream.
type Stream int

const (
	StreamIMU     Stream = iota // inertial measurements: acceleration + angular rate
	StreamOdom                  // body-frame odometry (e.g. wheel) velocity
	StreamAbsPose               // absolute pose observation (e.g. fiducial / GNSS attitude)
)

// AllStreams lists every supported stream in registration order.
var AllStreams = []Stream{StreamIMU, StreamOdom, StreamAbsPose}

func (s Stream) String() string {
	switch s {
	case StreamIMU:
		return "imu"
	case StreamOdom:
		return "odom"
	case StreamAbsPose:
		return "abspose"
	default:
		return "unknown"
	}
}

// Observation is a unified, timestamped sensor record in SI units.
//
// Time stamps are nanoseconds. NativeNS is the raw sensor/clock timestamp;
// LocalNS is the clock-corrected time on the engine reference clock and is
// the field used for ordering and state propagation.
type Observation struct {
	ID     int64
	Stream Stream
	// NativeNS is the timestamp as emitted by the sensor clock (may drift).
	NativeNS int64
	// LocalNS is the synchronised reference-clock timestamp. Zero means the
	// clock model was not usable yet; such records must never be consumed.
	LocalNS int64
	// IMU fields (m/s^2 and rad/s, body frame).
	Acc [3]float64
	Gyr [3]float64
	// Odometry field: body-frame linear velocity (m/s); angular velocity in
	// Gyr is reused when HasOdomW is true.
	OdomV    [3]float64
	HasOdomW bool
	// Absolute pose observation.
	HasAbsPose bool
	Pos        [3]float64
	Quat       [4]float64 // (w, x, y, z), Hamilton, q_WB
	// Measurement standard deviations (1-sigma), used to build R.
	SigmaPos             float64 // m
	SigmaAng             float64 // rad
	SigmaV               float64 // m/s
	SigmaIMUA, SigmaIMUG float64
	// Sequence number on the source stream (for gap/loss detection).
	Seq int64
	// ArrivalRefNS is the deterministic reference-clock time at which the
	// record is delivered (sensor time + link delay + possible reorder
	// delay). Used to fit the per-stream clock model; zero falls back to
	// LocalNS/NativeNS and therefore never blocks processing.
	ArrivalRefNS int64
}

// Time returns the synchronised timestamp as a time.Time.
func (o Observation) Time() time.Time { return time.Unix(0, o.LocalNS) }

// Decision describes how an observation was handled.
type Decision struct {
	ID       int64
	Stream   Stream
	LocalNS  int64
	Accepted bool
	Reason   Reason
	Detail   string
}

// Reason is a stable, distinguishable outcome code.
type Reason int

const (
	ReasonOK                 Reason = iota
	ReasonOutOfOrder                // arrived late but still inside the reorder window
	ReasonDroppedLate               // timestamp older than watermark, deterministically discarded
	ReasonDuplicate                 // identical (stream, timestamp) seen before
	ReasonTimeRegress               // timestamp older than a previously accepted one on the same stream
	ReasonClockNotReady             // clock model has insufficient data to map native -> local time
	ReasonOutOfWindow               // timestamp outside the reorder window
	ReasonBeforeStart               // earlier than engine start time
	ReasonGateReject                // failed the Mahalanobis / residual gate
	ReasonStale                     // stream exceeded its max-age budget
	ReasonDegraded                  // stream uncertainty inflated / sensor downgraded
	ReasonUnsupported               // unknown stream or missing payload
	ReasonSizeLimit                 // engine input size limit exceeded
	ReasonProcessingBudget          // per-step processing-time budget exceeded
	ReasonStreamInactive            // stream never produced usable data
	ReasonBufferBackpressure        // ordered buffer is full
)

func (r Reason) String() string {
	switch r {
	case ReasonOK:
		return "ok"
	case ReasonOutOfOrder:
		return "out_of_order_accepted"
	case ReasonDroppedLate:
		return "dropped_late_behind_watermark"
	case ReasonDuplicate:
		return "duplicate_timestamp"
	case ReasonTimeRegress:
		return "timestamp_regression"
	case ReasonClockNotReady:
		return "clock_model_not_ready"
	case ReasonOutOfWindow:
		return "outside_reorder_window"
	case ReasonBeforeStart:
		return "before_engine_start"
	case ReasonGateReject:
		return "mahalanobis_gate_reject"
	case ReasonStale:
		return "stream_stale"
	case ReasonDegraded:
		return "stream_degraded"
	case ReasonUnsupported:
		return "unsupported_observation"
	case ReasonSizeLimit:
		return "size_limit_exceeded"
	case ReasonProcessingBudget:
		return "processing_budget_exceeded"
	case ReasonStreamInactive:
		return "stream_inactive"
	case ReasonBufferBackpressure:
		return "buffer_backpressure"
	default:
		return "unknown_reason"
	}
}
