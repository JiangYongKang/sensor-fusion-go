package replay

import (
	"testing"

	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
)

func TestDeterministicGeneration(t *testing.T) {
	cfg := DefaultConfig()
	a := New(cfg).Generate()
	b := New(cfg).Generate()
	if len(a.Order) != len(b.Order) || len(a.Order) == 0 {
		t.Fatalf("lengths %d vs %d", len(a.Order), len(b.Order))
	}
	for i := range a.Order {
		x, y := a.Order[i], b.Order[i]
		if x != y {
			t.Fatalf("non-reproducible record at %d: %+v vs %+v", i, x, y)
		}
	}
	t.Logf("generated %d observations, %d truth points", len(a.Order), len(a.Truth))
}

func TestScenarioContents(t *testing.T) {
	cfg := DefaultConfig()
	res := New(cfg).Generate()
	var perStream [3]int
	lateArrivals := 0
	outliers := 0
	var lastMeasure [3]int64
	initialized := [3]bool{}
	for _, o := range res.Order {
		perStream[o.Stream]++
		if !initialized[o.Stream] {
			lastMeasure[o.Stream] = o.NativeNS
			initialized[o.Stream] = true
		} else if o.NativeNS < lastMeasure[o.Stream] {
			lateArrivals++
		}
		lastMeasure[o.Stream] = o.NativeNS
		if o.Stream == sensors.StreamAbsPose && o.Pos[1] > 3.0 {
			outliers++
		}
	}
	t.Logf("counts imu=%d odom=%d abs=%d lateDeliveries=%d injectedOutliers=%d",
		perStream[sensors.StreamIMU], perStream[sensors.StreamOdom], perStream[sensors.StreamAbsPose], lateArrivals, outliers)
	for s, n := range perStream {
		if n == 0 {
			t.Fatalf("stream %d has no data", s)
		}
	}
	if lateArrivals == 0 {
		t.Fatal("expected out-of-order deliveries")
	}
	if outliers == 0 {
		t.Fatal("expected injected pose outliers")
	}
	// Dropped frames: nominal IMU frames = duration/period.
	nominal := cfg.DurationNS / cfg.IMUPeriodNS
	if int64(perStream[sensors.StreamIMU]) >= nominal {
		t.Fatalf("drop frames missing: %d >= %d", perStream[sensors.StreamIMU], nominal)
	}
}

func TestClockDriftPresent(t *testing.T) {
	cfg := DefaultConfig()
	res := New(cfg).Generate()
	// Over 20 s at 8 ppm, IMU native-vs-local difference must reach tens of ms.
	var maxDiff int64
	for _, o := range res.Order {
		if o.Stream == sensors.StreamIMU {
			// local ~ native*(1+drift)+offset
			if d := o.ArrivalRefNS - o.NativeNS; abs(d) > maxDiff {
				maxDiff = abs(d)
			}
		}
	}
	t.Logf("max IMU native/reference gap ~%d ns (%.1f ms)", maxDiff, float64(maxDiff)/1e6)
	if maxDiff < 10_000_000 {
		t.Fatal("expected drift gap >10ms over the run")
	}
}

func abs(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}
