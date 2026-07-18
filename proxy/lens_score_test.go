package main

import "testing"

// Threshold resolution is fail-closed: only the selected model's calibrated
// values may drive interventions.
func TestLensThresholdResolution(t *testing.T) {
	var bare lensPerStepResult
	if _, _, ok := bare.calibratedThresholds(); ok {
		t.Fatal("missing calibration must not produce intervention thresholds")
	}
	withT := lensPerStepResult{Thresholds: &lensThresholds{OffRails: 0.6, Low: 0.45, Severe: 0.3}}
	low, severe, ok := withT.calibratedThresholds()
	if !ok || low != 0.45 || severe != 0.3 {
		t.Fatalf("calibrated thresholds = (%v, %v, %v)", low, severe, ok)
	}
	invalid := lensPerStepResult{Thresholds: &lensThresholds{Low: 0.2, Severe: 0.3}}
	if _, _, ok := invalid.calibratedThresholds(); ok {
		t.Fatal("severe > low must be rejected")
	}
}

// The same scores are interpreted only against the selected model's own
// calibration.
func TestAgentLensRegressionUsesPerModelThresholds(t *testing.T) {
	scores := []float64{0.40, 0.39}
	if _, fired := agentLensRegression(scores, 0.45, 0.30); !fired {
		t.Errorf("model-calibrated thresholds should fire on a 0.40/0.39 run")
	}
}

// Severe single-write short-circuit also honors the per-model severe value.
func TestAgentLensRegressionSevereIsPerModel(t *testing.T) {
	// One write at 0.32. Below a model severe of 0.35 → immediate fire.
	if _, fired := agentLensRegression([]float64{0.32}, 0.45, 0.35); !fired {
		t.Errorf("single write below per-model severe should fire")
	}
	// Same score under another valid calibration does not fire immediately.
	if _, fired := agentLensRegression([]float64{0.32}, 0.2, 0.1); fired {
		t.Errorf("single 0.32 write should not fire above calibrated severe=0.1")
	}
}
