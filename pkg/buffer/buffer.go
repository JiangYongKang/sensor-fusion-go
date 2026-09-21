// Package buffer provides a deterministic time-ordered ingress buffer.
//
// Observations may arrive out of order; the buffer holds them until their
// timestamp can be released in non-decreasing order. Duplicates (same
// stream + same local timestamp) and timestamps that regress past the
// release watermark are reported with distinguishable outcomes instead of
// being silently dropped or consumed early.
//
// Release rule (reorder delay): an observation at local time t becomes
// releasable once the drain time now satisfies now >= t + WindowNS. This
// deliberately waits one reorder window so late-arriving data can still be
// inserted ahead of the watermark and consumed in time order. The released
// watermark is the largest timestamp ever drained; inserts strictly behind
// it are deterministically rejected (they cannot be re-ordered into an
// already-propagated past).
//
// Tie-breaking at equal timestamps is fully deterministic: stream id, then
// source sequence number, then insertion order.
package buffer

import (
	"sort"

	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
)

// InsertOutcome reports what happened on Insert.
type InsertOutcome int

const (
	// InsertAccepted: stored in timestamp order (arrived in order).
	InsertAccepted InsertOutcome = iota
	// InsertAcceptedLate: arrived after a newer timestamp but still ahead of
	// the watermark; stored and will be consumed in time order.
	InsertAcceptedLate
	// InsertDuplicate: same (stream, LocalNS) already inserted.
	InsertDuplicate
	// InsertLate: behind the released watermark; cannot be consumed.
	InsertLate
	// InsertBackpressure: buffer capacity reached.
	InsertBackpressure
)

// Entry is an internal held record with bookkeeping.
type Entry struct {
	Obs     sensors.Observation
	Arrival int64 // insertion order index
}

// Buffer is a bounded reorder buffer keyed on Observation.LocalNS.
type Buffer struct {
	windowNS  int64
	capacity  int
	items     []Entry
	watermark int64
	seen      map[string]int64 // key -> LocalNS (== key timestamp)
	arrival   int64
	maxSeenNS int64 // largest timestamp ever inserted
}

// Config configures the reorder buffer.
type Config struct {
	WindowNS int64 // reorder delay: items release once now >= ts + WindowNS
	Capacity int   // max held observations
}

// New builds a buffer with safe defaults for a zero-value config.
func New(cfg Config) *Buffer {
	if cfg.Capacity <= 0 {
		cfg.Capacity = 4096
	}
	if cfg.WindowNS < 0 {
		cfg.WindowNS = 0
	}
	return &Buffer{
		windowNS:  cfg.WindowNS,
		capacity:  cfg.Capacity,
		watermark: -1,
		maxSeenNS: -1,
		seen:      make(map[string]int64),
	}
}

func key(o sensors.Observation) string {
	// Stream id + native timestamp uniquely identify a source sample; using
	// native time keeps duplicate detection stable even if a clock model is
	// re-fit. Engine additionally maps the outcome.
	return itoa(int64(o.Stream)) + ":" + itoa(o.NativeNS)
}

func itoa(x int64) string {
	if x == 0 {
		return "0"
	}
	neg := x < 0
	var b [24]byte
	i := len(b)
	if neg {
		x = -x
	}
	for x > 0 {
		i--
		b[i] = byte('0' + x%10)
		x /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// Insert stores o in time order. It never mutates or consumes the record.
func (b *Buffer) Insert(o sensors.Observation) InsertOutcome {
	k := key(o)
	if _, exists := b.seen[k]; exists {
		return InsertDuplicate
	}
	if o.LocalNS < b.watermark {
		return InsertLate
	}
	if len(b.items) >= b.capacity {
		return InsertBackpressure
	}
	outcome := InsertAccepted
	if o.LocalNS < b.maxSeenNS {
		outcome = InsertAcceptedLate
	}
	b.arrival++
	b.items = append(b.items, Entry{Obs: o, Arrival: b.arrival})
	b.seen[k] = o.LocalNS
	if o.LocalNS > b.maxSeenNS {
		b.maxSeenNS = o.LocalNS
	}
	sort.SliceStable(b.items, func(i, j int) bool {
		return lessEntry(b.items[i], b.items[j])
	})
	return outcome
}

func lessEntry(a, c Entry) bool {
	if a.Obs.LocalNS != c.Obs.LocalNS {
		return a.Obs.LocalNS < c.Obs.LocalNS
	}
	if a.Obs.Stream != c.Obs.Stream {
		return a.Obs.Stream < c.Obs.Stream
	}
	if a.Obs.Seq != c.Obs.Seq {
		return a.Obs.Seq < c.Obs.Seq
	}
	return a.Arrival < c.Arrival
}

// DrainReady removes and returns, in non-decreasing time order, every item
// releasable at nowNS (ts + WindowNS <= nowNS). The watermark advances to
// the largest released timestamp.
func (b *Buffer) DrainReady(nowNS int64) []sensors.Observation {
	var out []sensors.Observation
	for len(b.items) > 0 {
		head := b.items[0]
		if b.windowNS > 0 && nowNS < head.Obs.LocalNS+b.windowNS {
			break
		}
		if head.Obs.LocalNS > nowNS {
			break
		}
		out = append(out, head.Obs)
		if head.Obs.LocalNS > b.watermark {
			b.watermark = head.Obs.LocalNS
		}
		b.items = b.items[1:]
	}
	return out
}

// PopReady removes and returns the smallest-timestamp held item only when it
// is releasable at nowNS. When false is returned no state changes, which
// lets callers stop processing under a time budget without consuming or
// discarding the remaining items.
func (b *Buffer) PopReady(nowNS int64) (sensors.Observation, bool) {
	if len(b.items) == 0 {
		return sensors.Observation{}, false
	}
	head := b.items[0]
	if b.windowNS > 0 && nowNS < head.Obs.LocalNS+b.windowNS {
		return sensors.Observation{}, false
	}
	if head.Obs.LocalNS > nowNS {
		return sensors.Observation{}, false
	}
	b.items = b.items[1:]
	if head.Obs.LocalNS > b.watermark {
		b.watermark = head.Obs.LocalNS
	}
	return head.Obs, true
}

// PopNext removes and returns the smallest-timestamp item regardless of the
// reorder delay. Returns false when empty.
func (b *Buffer) PopNext() (sensors.Observation, bool) {
	if len(b.items) == 0 {
		return sensors.Observation{}, false
	}
	head := b.items[0]
	b.items = b.items[1:]
	if head.Obs.LocalNS > b.watermark {
		b.watermark = head.Obs.LocalNS
	}
	return head.Obs, true
}

// Peek returns the smallest-timestamp held item without removing it.
func (b *Buffer) Peek() (sensors.Observation, bool) {
	if len(b.items) == 0 {
		return sensors.Observation{}, false
	}
	return b.items[0].Obs, true
}

// Len is the number of held observations.
func (b *Buffer) Len() int { return len(b.items) }

// Watermark is the largest released timestamp (-1 before any release).
func (b *Buffer) Watermark() int64 { return b.watermark }

// Close releases retained data.
func (b *Buffer) Close() {
	b.items = nil
	b.seen = nil
}
