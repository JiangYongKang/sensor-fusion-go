package buffer

import (
	"testing"

	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
)

func obs(id int64, stream sensors.Stream, localNS int64) sensors.Observation {
	return sensors.Observation{ID: id, Stream: stream, NativeNS: localNS, LocalNS: localNS, Seq: id}
}

func TestOutOfOrderDrain(t *testing.T) {
	b := New(Config{WindowNS: 10, Capacity: 100})
	// Delivery order: 30, 10, 20, 40.
	seq := []sensors.Observation{obs(3, sensors.StreamIMU, 30), obs(1, sensors.StreamIMU, 10), obs(2, sensors.StreamIMU, 20), obs(4, sensors.StreamIMU, 40)}
	for _, o := range seq {
		got := b.Insert(o)
		t.Logf("insert ts=%d -> %v", o.LocalNS, got)
	}
	if got := b.Insert(obs(1, sensors.StreamIMU, 10)); got != InsertDuplicate {
		t.Fatalf("duplicate outcome = %v", got)
	}
	// At now=25: ts 10,20 releasable (20+10<=30? no) -> only with window 10:
	// ts10 release needs now>=20; ts20 needs now>=30.
	out := b.DrainReady(25)
	if len(out) != 1 || out[0].LocalNS != 10 {
		t.Fatalf("drain@25 = %v", out)
	}
	out = b.DrainReady(50)
	if len(out) != 3 {
		t.Fatalf("drain@50 len = %d", len(out))
	}
	for i := 1; i < len(out); i++ {
		if out[i].LocalNS < out[i-1].LocalNS {
			t.Fatalf("drain not ordered: %d then %d", out[i-1].LocalNS, out[i].LocalNS)
		}
	}
	t.Logf("released order: 10 %v %v %v", out[0].LocalNS, out[1].LocalNS, out[2].LocalNS)
	if b.Watermark() != 40 {
		t.Fatalf("watermark = %d", b.Watermark())
	}
	// ts 5 after watermark must be rejected late.
	if got := b.Insert(obs(99, sensors.StreamIMU, 5)); got != InsertLate {
		t.Fatalf("late insert outcome = %v", got)
	}
}

func TestLateArrivalBeforeWatermark(t *testing.T) {
	b := New(Config{WindowNS: 0, Capacity: 10})
	b.Insert(obs(1, sensors.StreamIMU, 100))
	b.Insert(obs(2, sensors.StreamIMU, 200))
	b.DrainReady(200)
	// Out-of-order frame ts=150 arrives after 200 was released.
	if got := b.Insert(obs(3, sensors.StreamIMU, 150)); got != InsertLate {
		t.Fatalf("expected InsertLate, got %v", got)
	}
}

func TestLateArrivalWithinWindowAccepted(t *testing.T) {
	b := New(Config{WindowNS: 50, Capacity: 10})
	b.Insert(obs(1, sensors.StreamIMU, 100))
	b.Insert(obs(2, sensors.StreamIMU, 200))
	if got := b.Insert(obs(3, sensors.StreamIMU, 150)); got != InsertAcceptedLate {
		t.Fatalf("expected InsertAcceptedLate, got %v", got)
	}
	out := b.DrainReady(1000)
	if len(out) != 3 || out[0].LocalNS != 100 || out[1].LocalNS != 150 || out[2].LocalNS != 200 {
		t.Fatalf("wrong order %v", out)
	}
}

func TestBackpressure(t *testing.T) {
	b := New(Config{WindowNS: 1000, Capacity: 3})
	for i, ts := range []int64{10, 20, 30} {
		if got := b.Insert(obs(int64(i), sensors.StreamIMU, ts)); got != InsertAccepted {
			t.Fatalf("insert %d = %v", ts, got)
		}
	}
	if got := b.Insert(obs(4, sensors.StreamIMU, 40)); got != InsertBackpressure {
		t.Fatalf("backpressure outcome = %v", got)
	}
}

func TestTieBreakDeterministic(t *testing.T) {
	b := New(Config{WindowNS: 0, Capacity: 10})
	// Same timestamp across streams: deterministic stream order regardless
	// of insertion order.
	a := obs(1, sensors.StreamAbsPose, 50)
	c := obs(2, sensors.StreamIMU, 50)
	d := obs(3, sensors.StreamOdom, 50)
	b.Insert(d)
	b.Insert(c)
	b.Insert(a)
	out := b.DrainReady(1000)
	if out[0].Stream != sensors.StreamIMU || out[1].Stream != sensors.StreamOdom || out[2].Stream != sensors.StreamAbsPose {
		t.Fatalf("tie order = %v %v %v", out[0].Stream, out[1].Stream, out[2].Stream)
	}
}
