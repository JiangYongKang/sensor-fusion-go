package timesync

import "testing"

func TestClockRecovery(t *testing.T) {
	cfg := Config{MinPairs: 4, MaxPairs: 16, MaxDrift: 1e-3}
	est := NewEstimator(0, cfg)

	const offsetNS int64 = 50_000
	const drift = 40e-6 // 40 ppm
	// Reference samples arrive every 10ms; native clock runs fast.
	var native []int64
	var ref []int64
	for k := 0; k < 12; k++ {
		r := int64(k) * 10_000_000
		nf := (float64(r) - float64(offsetNS)) / (1 + drift)
		n := int64(nf + 0.5)
		ref = append(ref, r)
		native = append(native, n)
	}
	for i := 0; i < 3; i++ {
		est.AddSample(native[i], ref[i])
	}
	if m := est.Model(); m.Ready {
		t.Fatal("clock should not be ready before MinPairs")
	}
	if _, ok := est.Model().ToLocal(native[0]); ok {
		t.Fatal("ToLocal must fail while not ready")
	}
	for i := 3; i < len(native); i++ {
		est.AddSample(native[i], ref[i])
	}
	m := est.Model()
	if !m.Ready {
		t.Fatal("clock model should be ready")
	}
	t.Logf("fitted drift=%.6e offsetNS=%.3f (truth drift=%.6e offset=%d)",
		m.Drift, m.OffsetNS, drift, offsetNS)
	bad := 0
	for i := range native {
		got, _ := m.ToLocal(native[i])
		if d := got - ref[i]; d > 200 || d < -200 {
			bad++
			t.Errorf("sample %d: local=%d ref=%d err=%d ns", i, got, ref[i], d)
		}
	}
	if bad > 2 {
		t.Fatalf("too many inaccurate corrections: %d", bad)
	}
}

func TestDeterminismAndRing(t *testing.T) {
	run := func() [2]float64 {
		e := NewEstimator(1, Config{MinPairs: 2, MaxPairs: 5, MaxDrift: 1e-3})
		for k := 0; k < 20; k++ {
			e.AddSample(int64(k)*1_000_000, int64(k)*1_000_000+100)
		}
		m := e.Model()
		return [2]float64{m.OffsetNS, m.Drift}
	}
	a, b := run(), run()
	if a != b {
		t.Fatalf("non-deterministic fit: %v vs %v", a, b)
	}
}

func TestRejectsImplausibleDrift(t *testing.T) {
	e := NewEstimator(0, Config{MinPairs: 2, MaxPairs: 8, MaxDrift: 1e-3})
	e.AddSample(0, 0)
	e.AddSample(1_000_000, 10_000_000) // 900% drift
	if e.Model().Ready {
		t.Fatal("implausible drift must not produce a ready model")
	}
}
