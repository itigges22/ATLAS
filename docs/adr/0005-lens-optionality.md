# ADR 0005: Lens is optional and calibration-gated

Status: accepted (V3.1.2 calibration work)

## Context
Lens scores are model-relative. Uncalibrated or missing artifacts must
not steer selection with wrong numbers.

## Decision
Three explicit states, all surfaced identically in health, doctor,
lens check, and the TUI badge: loaded+calibrated (full interventions),
loaded+uncalibrated (raw telemetry, neutral normalized scores,
threshold interventions disabled), and disabled (identity/dim mismatch
or missing artifacts — scoring returns explicit unavailable flags).
Scoring never silently substitutes defaults for calibrated output.

## Consequences
A fresh install without `atlas lens build` runs uncalibrated and says
so everywhere. The router's energy signal logs once per load cycle when
disabled. V3 candidate selection degrades to neutral-score ordering.
