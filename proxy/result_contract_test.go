package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// Stage B contract tests. These pin the FACTS a ToolResult reports, before
// any consumer is changed.
//
// The defect they guard against is the one the locked benchmark found:
// `Success` conflated "the tool ran", "bytes changed" and "the content is
// valid", so a warned invalid write and a validated one were
// indistinguishable, and a successful delete looked like productive work.
//
// Every new field must FAIL CLOSED at its zero value. A result that was
// never populated, or that carries a value this build does not recognise,
// must never read as applied or as passed.

func TestZeroValueResultIsNeitherAppliedNorValidated(t *testing.T) {
	var r ToolResult
	if r.MutationStatus.Applied() {
		t.Error("zero-value MutationStatus reads as applied")
	}
	if r.ValidationStatus.Passed() {
		t.Error("zero-value ValidationStatus reads as passed")
	}
	if r.ValidationKind != ValidationKindNone {
		t.Errorf("zero-value ValidationKind = %q, want none", r.ValidationKind)
	}
	if r.MutationStatus != MutationNone {
		t.Errorf("zero-value MutationStatus = %q, want none", r.MutationStatus)
	}
	if r.ValidationStatus != ValidationNotRun {
		t.Errorf("zero-value ValidationStatus = %q, want not_run", r.ValidationStatus)
	}
}

// An unrecognised value is not a success. This matters across a version skew
// where one side emits a status the other does not know.
func TestUnknownEnumValuesAreNotSuccess(t *testing.T) {
	for _, v := range []MutationStatus{"", "APPLIED", "applied_maybe", "yes", "unknown"} {
		if v == MutationApplied {
			continue
		}
		if v.Applied() {
			t.Errorf("MutationStatus(%q) reads as applied", v)
		}
	}
	for _, v := range []ValidationStatus{"", "PASSED", "passed_ish", "ok", "unknown"} {
		if v == ValidationPassed {
			continue
		}
		if v.Passed() {
			t.Errorf("ValidationStatus(%q) reads as passed", v)
		}
	}
}

// not_run and not_applicable are distinct and neither is passed. Collapsing
// them is how "we could not check" silently becomes "we checked".
func TestNotRunAndNotApplicableAreNotPassed(t *testing.T) {
	if ValidationNotRun.Passed() {
		t.Error("not_run reads as passed")
	}
	if ValidationNotApplicable.Passed() {
		t.Error("not_applicable reads as passed")
	}
	if ValidationFailed.Passed() {
		t.Error("failed reads as passed")
	}
	if ValidationNotRun == ValidationNotApplicable {
		t.Error("not_run and not_applicable must stay distinguishable")
	}
}

// Legacy wire compatibility: `success` keeps its meaning and its shape, and
// the new fields are additive and omitted when unset.
func TestLegacyWireSuccessUnchanged(t *testing.T) {
	b, err := json.Marshal(ToolResult{Success: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"success":true`) {
		t.Fatalf("legacy success field changed shape: %s", b)
	}
	for _, unwanted := range []string{"mutation_status", "validation_status", "validation_kind"} {
		if strings.Contains(string(b), unwanted) {
			t.Errorf("unset %s must be omitted from the wire, got: %s", unwanted, b)
		}
	}
}

func TestNewFieldsRoundTrip(t *testing.T) {
	in := ToolResult{
		Success:          false,
		MutationStatus:   MutationRefused,
		ValidationKind:   ValidationKindSyntax,
		ValidationStatus: ValidationFailed,
		ValidationDetail: "SyntaxError: invalid syntax (line 4)",
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out ToolResult
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if out.MutationStatus != MutationRefused || out.ValidationStatus != ValidationFailed ||
		out.ValidationKind != ValidationKindSyntax || out.ValidationDetail != in.ValidationDetail {
		t.Fatalf("round trip lost fields: %+v", out)
	}
}

// A payload from an older producer carries no new fields at all. It must
// decode to the fail-closed zero values rather than to applied/passed.
func TestLegacyPayloadDecodesFailClosed(t *testing.T) {
	var out ToolResult
	if err := json.Unmarshal([]byte(`{"success":true}`), &out); err != nil {
		t.Fatal(err)
	}
	if out.MutationStatus.Applied() || out.ValidationStatus.Passed() {
		t.Fatalf("a legacy payload decoded as applied/validated: %+v", out)
	}
}

// Syntax validation is labelled as such. It is not evidence that the task
// was verified, and the kind is what keeps those separable.
func TestSyntaxPassIsLabelledSyntaxNotTaskVerification(t *testing.T) {
	r := ToolResult{
		Success: true, MutationStatus: MutationApplied,
		ValidationKind: ValidationKindSyntax, ValidationStatus: ValidationPassed,
	}
	if r.ValidationKind != ValidationKindSyntax {
		t.Fatalf("ValidationKind = %q, want syntax", r.ValidationKind)
	}
	if !r.ValidationStatus.Passed() {
		t.Fatal("an explicit syntax pass must read as passed")
	}
}
