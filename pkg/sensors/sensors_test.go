package sensors

import "testing"

func TestReasonsAreDistinct(t *testing.T) {
	seen := map[string]Reason{}
	for r := Reason(0); r <= ReasonBufferBackpressure; r++ {
		name := r.String()
		if other, dup := seen[name]; dup {
			t.Fatalf("reason %d shares string %q with %d", r, name, other)
		}
		seen[name] = r
		t.Logf("reason %2d -> %s", r, name)
	}
}

func TestStreamNames(t *testing.T) {
	want := map[Stream]string{StreamIMU: "imu", StreamOdom: "odom", StreamAbsPose: "abspose"}
	for s, name := range want {
		if s.String() != name {
			t.Fatalf("%d -> %q want %q", s, s.String(), name)
		}
	}
}
