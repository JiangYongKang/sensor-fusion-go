package fusion

import "time"

// alignParamsFromConfig 从 Config 派生时钟对齐参数并填充默认值。
func alignParamsFromConfig(cfg Config) alignParams {
	enabled := true
	if cfg.AlignEnabled != nil {
		enabled = *cfg.AlignEnabled
	}
	minSamples := cfg.AlignMinSamples
	if minSamples <= 0 {
		minSamples = 30
	}
	confirm := cfg.AlignConfirmSamples
	if confirm <= 0 {
		confirm = 10
	}
	activateAt := cfg.AlignActivateAt
	if activateAt <= 0 {
		activateAt = 50 * time.Millisecond
	}
	maxOffset := cfg.AlignMaxOffset
	if maxOffset <= 0 {
		maxOffset = 5 * time.Second
	}
	maxSkew := cfg.AlignMaxSkew
	if maxSkew <= 0 {
		maxSkew = 5000e-6 // 0.5%
	}
	lambda := cfg.AlignForgetting
	if lambda <= 0 || lambda > 1 {
		lambda = 0.99
	}
	initBStd := cfg.AlignInitStd
	if initBStd <= 0 {
		initBStd = 5 * time.Second
	}
	qB := cfg.AlignProcessStd
	if qB <= 0 {
		qB = 5e-4 // 0.5ms/√s 随机游走
	}
	measStd := cfg.AlignMeasStd
	if measStd <= 0 {
		measStd = 20 * time.Millisecond
	}
	// qS 直接由偏移随机游走与一个温和的频偏游走派生（约 1ppm/√s）。
	const qS = 1e-6 / 1.0
	maxHeld := cfg.AlignMaxHeld
	if maxHeld <= 0 {
		maxHeld = 30
	}
	minLin := cfg.AlignMinLinVel
	if minLin <= 0 {
		minLin = 0.05
	}
	minAng := cfg.AlignMinAngVel
	if minAng <= 0 {
		minAng = 0.02
	}
	return alignParams{
		enabled:            enabled,
		requireCalibration: cfg.AlignRequireCalibration,
		minSamples:         minSamples,
		confirmSamples:     confirm,
		activateAt:         activateAt.Seconds(),
		maxOffset:          maxOffset.Seconds(),
		maxSkew:            maxSkew,
		forgetting:         lambda,
		initBStd:           initBStd.Seconds(),
		initSStd:           0.01,
		qB:                 qB,
		qS:                 qS,
		maxHeld:            maxHeld,
		learnGate:          maxOffset.Seconds(),
		bootstrapGate:      maxOffset.Seconds() / 2,
		measStd:            measStd.Seconds(),
		measVarFloor:       measStd.Seconds() * measStd.Seconds(),
		robustSigma:        6.0,
		minLinVel:          minLin,
		minAngVel:          minAng,
		skewActivate:       100e-6, // 100ppm
		candAlpha:          0.05,
	}
}
