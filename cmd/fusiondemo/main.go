// Command fusiondemo runs the local sensor-fusion engine against a
// deterministic replay (out-of-order delivery, dropped frames, injected
// outliers, clock drift and long runtime) and prints a reproducible report.
//
// Usage:
//
//	go run ./cmd/fusiondemo [-duration 20s] [-seed 20260921] [-verbose]
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/highcumontoa/sensor-fusion-go/pkg/engine"
	"github.com/highcumontoa/sensor-fusion-go/pkg/fusion"
	"github.com/highcumontoa/sensor-fusion-go/pkg/pose"
	"github.com/highcumontoa/sensor-fusion-go/pkg/replay"
	"github.com/highcumontoa/sensor-fusion-go/pkg/sensors"
	"github.com/highcumontoa/sensor-fusion-go/pkg/timesync"
)

func main() {
	duration := flag.Duration("duration", 20*time.Second, "replay duration")
	seed := flag.Int64("seed", 20260921, "deterministic RNG seed")
	verbose := flag.Bool("verbose", false, "print every non-OK decision")
	samplesOnly := flag.Bool("samples", false, "print 3 sample decisions per reason and exit")
	flag.Parse()

	cfg := replay.DefaultConfig()
	cfg.Seed = *seed
	cfg.DurationNS = int64(*duration / time.Nanosecond)

	data := replay.New(cfg).Generate()
	gen := replay.New(cfg)

	ecfg := engine.Config{StartNS: 0}
	ecfg.Buffer.WindowNS = 80_000_000
	ecfg.Buffer.Capacity = 16_384
	ecfg.Clock = map[sensors.Stream]timesync.Config{}
	ecfg.LinkDelayNS = map[sensors.Stream]int64{}
	for _, s := range sensors.AllStreams {
		ecfg.Clock[s] = timesync.Config{MinPairs: 2, MaxPairs: 64, MaxDrift: 1e-3}
		ecfg.LinkDelayNS[s] = cfg.LinkDelayNS
	}
	ecfg.Fusion = fusion.Config{
		InitialPosSigma: 1, InitialVelSigma: 1, InitialAttSigma: 0.2,
		Process:         fusion.Noise{Accel: 0.08, Gyro: 2e-3},
		Gravity:         [3]float64{0, 0, -9.81},
		MaxPositionJump: 8,
		GateSigma:       4,
		DefaultSigmaPos: 0.02,
		DefaultSigmaAng: 0.01,
		DefaultSigmaV:   0.03,
		MinEigenvalue:   1e-12,
	}
	ecfg.Degrade = map[sensors.Stream]engine.DegradeRule{
		sensors.StreamIMU:     {DegradeAgeNS: 100_000_000, StaleAgeNS: 1_000_000_000, SigmaFactor: 4},
		sensors.StreamOdom:    {DegradeAgeNS: 250_000_000, StaleAgeNS: 2_000_000_000, SigmaFactor: 4},
		sensors.StreamAbsPose: {DegradeAgeNS: 600_000_000, StaleAgeNS: 3_000_000_000, SigmaFactor: 4},
	}
	ecfg.Limits = engine.Limits{MaxPending: 16_384, MaxObservations: 5_000_000}

	e := engine.New(ecfg)
	defer e.Close()

	reasonCounts := map[sensors.Reason]int{}
	var maxPosErr, maxAttErr float64
	var sumPos, sumAtt float64
	var nEval int

	ctx := context.Background()
	startWall := time.Now()
	for i, o := range data.Order {
		rep := e.Submit(ctx, o)
		reasonCounts[rep.Decision.Reason]++
		for _, d := range rep.Released {
			reasonCounts[d.Reason]++
		}
		// Advance the release clock to the current delivery time so frames
		// past their reorder window are consumed.
		now := o.ArrivalRefNS
		if now == 0 {
			now = o.LocalNS
		}
		ds := e.AdvanceTo(ctx, now)
		for _, d := range ds {
			reasonCounts[d.Reason]++
		}
		// Sample a few decisions for the human-readable trail.
		if (*verbose || *samplesOnly) && reasonCounts[rep.Decision.Reason] <= 3 {
			fmt.Printf("[in %5d] id=%-4d %-7s native=%d reason=%s detail=%s\n",
				i, o.ID, o.Stream, o.NativeNS, rep.Decision.Reason, rep.Decision.Detail)
		}
		// Periodic evaluation against analytic truth.
		if i%199 == 0 {
			st := e.Snapshot()
			if st.TNS > 0 {
				tr := gen.TruthAt(st.TNS)
				pe := pose.Vec3{X: tr.Pos[0], Y: tr.Pos[1], Z: tr.Pos[2]}.
					Sub(st.Pos).Norm()
				ae := st.Quat.AngleTo(pose.Quat{W: tr.Quat[0], X: tr.Quat[1], Y: tr.Quat[2], Z: tr.Quat[3]})
				maxPosErr = math.Max(maxPosErr, pe)
				maxAttErr = math.Max(maxAttErr, ae)
				sumPos += pe
				sumAtt += ae
				nEval++
			}
		}
	}
	// Final flush well beyond the reorder window.
	final := data.Order[len(data.Order)-1].ArrivalRefNS + 1_000_000_000
	for _, d := range e.AdvanceTo(ctx, final) {
		reasonCounts[d.Reason]++
	}
	wall := time.Since(startWall)

	st := e.Snapshot()
	tr := gen.TruthAt(st.TNS)
	finalPE := pose.VecFrom(tr.Pos).Sub(st.Pos).Norm()
	qtr := pose.Normalize(pose.Quat{W: tr.Quat[0], X: tr.Quat[1], Y: tr.Quat[2], Z: tr.Quat[3]})
	finalAttErr := st.Quat.AngleTo(qtr)

	fmt.Println("================ fusion replay report ================")
	fmt.Printf("seed=%d duration=%s observations=%d wall=%s\n", *seed, duration, len(data.Order), wall.Round(time.Millisecond))
	fmt.Printf("accepted=%d rejected=%d filterSteps=%d finalTime=%s\n",
		e.AcceptedCount(), e.RejectedCount(), st.Step, time.Duration(st.TNS))
	var reasons []sensors.Reason
	for r := range reasonCounts {
		reasons = append(reasons, r)
	}
	sort.Slice(reasons, func(i, j int) bool { return reasons[i] < reasons[j] })
	for _, r := range reasons {
		fmt.Printf("  decision %-32s %d\n", r, reasonCounts[r])
	}
	fmt.Println("------------------------------------------------------")
	for _, h := range e.Health() {
		fmt.Printf("health %-7s mode=%-11s samples=%-5d age=%s\n",
			h.Stream, h.Mode, h.Samples, time.Duration(h.AgeNS))
	}
	fmt.Println("------------------------------------------------------")
	if nEval > 0 {
		fmt.Printf("truth error: meanPos=%.4fm maxPos=%.4fm meanAtt=%.5frad maxAtt=%.5frad\n",
			sumPos/float64(nEval), maxPosErr, sumAtt/float64(nEval), maxAttErr)
	}
	fmt.Printf("final error: pos=%.4fm att=%.5frad |q|=%.12f covSPD=",
		finalPE, finalAttErr, st.Quat.Norm())
	if _, ok := st.Cov.Cholesky(); ok {
		fmt.Println("true")
	} else {
		fmt.Println("FALSE")
	}
	os.Stdout.Sync()
}
