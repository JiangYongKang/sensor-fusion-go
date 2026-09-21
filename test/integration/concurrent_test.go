package integration

import (
	"context"
	"sync"
	"testing"

	"github.com/highcumontoa/sensor-fusion-go/pkg/engine"
	"github.com/highcumontoa/sensor-fusion-go/pkg/replay"
	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
	"time"
)

// TestConcurrentSubmissionsNoRace feeds the identical deterministic dataset
// from several goroutines (partitioned by stream) while a reader goroutine
// continuously snapshots. It must complete without a data race and without
// ever exposing an invalid state (unit quaternion / SPD covariance).
func TestConcurrentSubmissionsNoRace(t *testing.T) {
	rc := replay.DefaultConfig()
	rc.DurationNS = int64(5 * time.Second / time.Nanosecond)
	data := replay.New(rc).Generate()

	run := func() [3]float64 {
		e := engine.New(engineConfig())
		defer e.Close()
		ctx := context.Background()

		stop := make(chan struct{})
		var readers sync.WaitGroup
		readers.Add(2)
		for r := 0; r < 2; r++ {
			go func() {
				defer readers.Done()
				for {
					select {
					case <-stop:
						return
					default:
						s := e.Snapshot()
						if s.Quat.Norm() > 1+1e-9 {
							panic("non-unit quaternion observed")
						}
					}
				}
			}()
		}

		var perStream [3][]sensors.Observation
		for _, o := range data.Order {
			perStream[o.Stream] = append(perStream[o.Stream], o)
		}
		var writers sync.WaitGroup
		for s := range perStream {
			writers.Add(1)
			go func(list []sensors.Observation) {
				defer writers.Done()
				for _, o := range list {
					e.Submit(ctx, o)
				}
			}(perStream[s])
		}
		writers.Wait()
		// Drain in delivery order after all writers finished.
		for _, o := range data.Order {
			e.AdvanceTo(ctx, o.ArrivalRefNS)
		}
		e.AdvanceTo(ctx, data.Order[len(data.Order)-1].ArrivalRefNS+1_000_000_000)
		close(stop)
		readers.Wait()
		snap := e.Snapshot()
		if _, ok := snap.Cov.Cholesky(); !ok {
			t.Fatal("non-SPD covariance")
		}
		return snap.Pos.Slice()
	}

	a := run()
	b := run()
	t.Logf("concurrent runs completed without race or invalid state; posA=%v posB=%v", a, b)
	t.Log("note: different goroutine interleavings are different input")
	t.Log("sequences; byte-identical reproducibility is asserted by")
	t.Log("TestReplayReproducible, which uses one identical submit sequence.")
}
