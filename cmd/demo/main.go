// demo 使用本地生成的回放数据运行融合引擎，打印判定统计与末端误差。
package main

import (
	"fmt"
	"time"

	fusion "github.com/highcumontoa/sensor-fusion-go/fusion"
)

func main() {
	cfg := fusion.ReplayConfig{
		Seed: 7, Duration: 60 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.05, OutlierProb: 0.02, OutlierScale: 5,
		ReorderWindow: 16, ClockDrift: 50 * time.Millisecond,
		PoseGap: 5 * time.Second,
	}
	obs := fusion.GenerateReplay(cfg)
	fmt.Printf("回放: %d 条观测 / %v\n", len(obs), cfg.Duration)

	eng := fusion.NewEngine(fusion.Config{
		ProcessNoise: [9]float64{0.001, 0.001, 0.001, 0.05, 0.05, 0.05, 0.001, 0.001, 0.001},
	})
	const batchSize = 64
	for i := 0; i < len(obs); i += batchSize {
		end := i + batchSize
		if end > len(obs) {
			end = len(obs)
		}
		eng.Ingest(obs[i:end])
	}
	end := obs[0].Stamp
	for _, o := range obs {
		if o.Stamp.After(end) {
			end = o.Stamp
		}
	}
	eng.AdvanceTo(end)

	stat := make(map[fusion.RejectReason]int)
	for _, d := range eng.Decisions() {
		stat[d.Reason]++
	}
	fmt.Println("判定分布:", stat)

	s, _, ok := eng.Snapshot()
	if !ok {
		fmt.Println("引擎未初始化")
		return
	}
	traj := fusion.DefaultTrajectory(obs[0].Stamp, cfg.Duration)
	truth := traj.TruePoseAt(s.Stamp)
	fmt.Printf("末端状态: pos=%+.3f vel=%+.3f\n", s.Position, s.Velocity)
	fmt.Printf("真值    : pos=%+.3f\n", truth.Position)
}
