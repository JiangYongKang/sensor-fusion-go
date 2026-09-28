// demo 使用本地生成的回放数据运行融合引擎，打印判定统计与末端误差。
package main

import (
	"fmt"
	"time"

	fusion "github.com/highcumontoa/sensor-fusion-go/fusion"
)

func main() {
	// 各路独立、随时间变化的设备时钟漂移：位姿路先快后慢，里程计反向。
	clocks := map[fusion.SensorKind]fusion.ClockProfile{
		fusion.SensorAbsolutePose: {
			{At: 0, Offset: 60 * time.Millisecond, Skew: 0},
			{At: 10 * time.Second, Offset: 60 * time.Millisecond, Skew: 400e-6},
			{At: 13 * time.Second, Offset: 132 * time.Millisecond, Skew: 0},
			{At: 22 * time.Second, Offset: 132 * time.Millisecond, Skew: -350e-6},
			{At: 25 * time.Second, Offset: 69 * time.Millisecond, Skew: 0},
		},
		fusion.SensorOdometry: {
			{At: 0, Offset: -40 * time.Millisecond, Skew: 0},
		},
	}
	cfg := fusion.ReplayConfig{
		Seed: 7, Duration: 60 * time.Second,
		IMURate: 100, OdomRate: 20, PoseRate: 5,
		DropProb: 0.05, OutlierProb: 0.02, OutlierScale: 5,
		ReorderWindow: 16, Clocks: clocks,
		PoseGap: 5 * time.Second,
	}
	obs := fusion.GenerateReplay(cfg)
	fmt.Printf("回放: %d 条观测 / %v（含各路时变时钟漂移）\n", len(obs), cfg.Duration)

	ec := fusion.Config{
		ProcessNoise: [9]float64{0.001, 0.001, 0.001, 0.05, 0.05, 0.05, 0.001, 0.001, 0.001},
	}
	eng := fusion.NewEngine(ec)
	const batchSize = 64
	for i := 0; i < len(obs); i += batchSize {
		end := i + batchSize
		if end > len(obs) {
			end = len(obs)
		}
		eng.Ingest(obs[i:end])
	}
	// 统一时间轴以 IMU 为基准，推进到最晚 IMU 读数。
	var end time.Time
	for _, o := range obs {
		if o.Kind == fusion.SensorIMU && o.Stamp.After(end) {
			end = o.Stamp
		}
	}
	eng.AdvanceTo(end)

	stat := make(map[fusion.RejectReason]int)
	for _, d := range eng.Decisions() {
		stat[d.Reason]++
	}
	fmt.Println("判定分布:", stat)
	fmt.Println("在线时钟对齐状态:")
	for _, k := range []fusion.SensorKind{fusion.SensorIMU, fusion.SensorOdometry, fusion.SensorAbsolutePose} {
		st := eng.AlignStatuses()[k]
		fmt.Printf("  %-14s phase=%-11s mapped=%-5v offset=%v skew=%.0fppm\n",
			k, st.Phase, st.Mapped, st.Offset, st.Skew*1e6)
	}

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
