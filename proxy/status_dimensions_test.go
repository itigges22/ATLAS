package main

import "testing"

func dimByName(dims []StatusDimension, name string) StatusDimension {
	for _, d := range dims {
		if d.Name == name {
			return d
		}
	}
	return StatusDimension{}
}

func TestBuildDimensionsSevenRows(t *testing.T) {
	dims := buildDimensions(LensStatus{}, ASAStatus{Verdict: "missing"})
	want := []string{"model_runtime", "direct_agent", "lens_identity",
		"lens_scoring", "lens_calibration", "lens_intervention", "asa"}
	if len(dims) != len(want) {
		t.Fatalf("expected %d dimensions, got %d", len(want), len(dims))
	}
	for i, n := range want {
		if dims[i].Name != n {
			t.Errorf("dimension %d = %q, want %q", i, dims[i].Name, n)
		}
	}
}

func TestDirectAgentAlwaysSupported(t *testing.T) {
	// Even with a fully disabled lens, the direct agent is model-agnostic.
	dims := buildDimensions(LensStatus{Verdict: "unreachable"},
		ASAStatus{Verdict: "missing"})
	if d := dimByName(dims, "direct_agent"); d.Status != "supported" {
		t.Fatalf("direct_agent should always be supported, got %q", d.Status)
	}
}

func TestInterventionNeutralWhenUncalibrated(t *testing.T) {
	// Loaded + scoring available but NOT calibrated → intervention must
	// be "neutral" (never "active"), matching the runtime guarantee.
	lens := LensStatus{
		Verdict: "uncalibrated", CostFieldLoaded: true, GxLoaded: true,
		CostFieldDim: 3840, EmbedDim: 3840,
		CxCalibrated: false, GxCalibrated: false,
	}
	dims := buildDimensions(lens, ASAStatus{Verdict: "missing"})
	if d := dimByName(dims, "lens_calibration"); d.Status != "uncalibrated" {
		t.Errorf("calibration = %q, want uncalibrated", d.Status)
	}
	if d := dimByName(dims, "lens_intervention"); d.Status != "neutral" {
		t.Fatalf("intervention = %q, want neutral when uncalibrated", d.Status)
	}
}

func TestInterventionActiveOnlyWhenCalibrated(t *testing.T) {
	lens := LensStatus{
		Verdict: "supported", CostFieldLoaded: true, GxLoaded: true,
		CostFieldDim: 3840, EmbedDim: 3840,
		CxCalibrated: true, GxCalibrated: true,
	}
	dims := buildDimensions(lens, ASAStatus{Verdict: "supported"})
	if d := dimByName(dims, "lens_intervention"); d.Status != "active" {
		t.Fatalf("intervention = %q, want active when calibrated", d.Status)
	}
}

func TestInterventionDisabledWhenNoArtifacts(t *testing.T) {
	dims := buildDimensions(LensStatus{Verdict: "no-artifacts"},
		ASAStatus{Verdict: "missing"})
	if d := dimByName(dims, "lens_intervention"); d.Status != "disabled" {
		t.Fatalf("intervention = %q, want disabled with no artifacts", d.Status)
	}
}

func TestDimMismatchSurfaced(t *testing.T) {
	lens := LensStatus{
		Verdict: "dim-mismatch", CostFieldLoaded: true,
		CostFieldDim: 4096, EmbedDim: 3840,
	}
	dims := buildDimensions(lens, ASAStatus{Verdict: "missing"})
	if d := dimByName(dims, "lens_identity"); d.Status != "dim-mismatch" {
		t.Fatalf("identity = %q, want dim-mismatch", d.Status)
	}
}
