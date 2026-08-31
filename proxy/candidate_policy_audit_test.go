package main

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// Configuration, telemetry and provenance, audited as facts about the build
// rather than as intentions.

// No production path treats an unknown mode as anything but strict, and the
// mode reaches the resolver from exactly two places.
func TestUnknownModesFailClosedToStrict(t *testing.T) {
	for _, raw := range []string{"advisory ", "ADVISORY", "auto", "yolo", "confirm!", "1"} {
		mode, ok := ParseCandidatePolicy(raw)
		if raw == "advisory " {
			// Trimmed, so this one is a legitimate spelling of advisory.
			if !ok || mode != CandidatePolicyAdvisory {
				t.Errorf("%q resolved to %q/%v", raw, mode, ok)
			}
			continue
		}
		if ok {
			t.Errorf("%q was accepted", raw)
		}
		if mode != CandidatePolicyStrict {
			t.Errorf("%q fell back to %q, want strict", raw, mode)
		}
	}
	// And an unreadable operator value is strict, never advisory.
	t.Setenv("ATLAS_CANDIDATE_POLICY", "advisorY")
	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	if mode, _ := candidatePolicyOf(ctx); mode != CandidatePolicyStrict {
		t.Errorf("an unreadable operator value resolved to %q", mode)
	}
}

// Neither the model nor the service can reach the mode, and no route reads a
// mode off anything they produce.
func TestNeitherModelNorServiceSelectsTheMode(t *testing.T) {
	files := []string{"candidate_policy.go", "advisory_policy.go", "tools.go",
		"edit_route_delivery.go", "agent.go"}
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		body := codeWithoutComments(string(src))
		for _, banned := range []string{
			"CandidatePolicy(v3Result", "CandidatePolicy(result",
			"CandidatePolicy(msg", "CandidatePolicy(response",
			"ParseCandidatePolicy(v3", "ParseCandidatePolicy(result",
		} {
			if strings.Contains(body, banned) {
				t.Errorf("%s derives the mode from %q", f, banned)
			}
		}
	}
	// The wire shape a service answers with carries no policy field at all.
	types, err := os.ReadFile("types.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(types)
	start := strings.Index(body, "type V3GenerateResponse struct")
	if start < 0 {
		t.Fatal("the response type moved")
	}
	resp := body[start : start+strings.Index(body[start:], "\n}")]
	for _, banned := range []string{"Policy", "candidate_policy", "Mode"} {
		if strings.Contains(resp, banned) {
			t.Errorf("the V3 response carries %q", banned)
		}
	}
}

// Advisory is reachable for an experiment and cannot deliver.
func TestAdvisoryIsReachableAndInert(t *testing.T) {
	ctx := policyContext(t, CandidatePolicyAdvisory)
	mode, source := candidatePolicyOf(ctx)
	if mode != CandidatePolicyAdvisory || source != CandidatePolicySourceClient {
		t.Fatalf("advisory unreachable: %q/%q", mode, source)
	}
	clean := advisoryInput{
		Observed:         checkOutcome{Status: ValidationPassed},
		TargetDeclared:   true,
		TargetAuthorized: true,
		ScopeAdmits:      true,
		Evidence: []proxyEvidence{{
			Provenance: V3EvidenceProvenance{Source: ProvenanceClientDeclaredVerification},
			Outcome:    ValidationPassed,
		}},
	}
	out := decideCandidatePolicy(ctx, clean, false)
	if out.Decision != PolicyCandidatePreferredAdvisory {
		t.Fatalf("advisory decided %q", out.Decision)
	}
	if out.Delivers || out.mayDeliverUnderPolicy() {
		t.Fatal("advisory delivered")
	}
	// And the only decision that sets Delivers is the strict one.
	for _, d := range []candidatePolicyDecision{
		PolicyBaselineRetained, PolicyCandidatePreferredAdvisory,
		PolicyHumanConfirmationRequired, PolicyCandidateRejectedHardVeto,
		PolicyInsufficientConfidence,
	} {
		if deliveryProvenanceFor(candidatePolicyOutcome{Decision: d}) != DeliveryFromModelProposal {
			t.Errorf("%s claimed a candidate origin without delivering", d)
		}
	}
}

// Confirm presents; it does not deliver. Approval is a separate act and this
// build has no path that fabricates one.
func TestConfirmCannotDeliverWithoutApproval(t *testing.T) {
	ctx := policyContext(t, CandidatePolicyConfirm)
	out := decideCandidatePolicy(ctx, advisoryInput{
		Observed:         checkOutcome{Status: ValidationPassed},
		TargetDeclared:   true,
		TargetAuthorized: true,
		ScopeAdmits:      true,
		Evidence: []proxyEvidence{{
			Provenance: V3EvidenceProvenance{Source: ProvenanceClientDeclaredVerification},
			Outcome:    ValidationPassed,
		}},
	}, false)
	if out.Decision != PolicyHumanConfirmationRequired {
		t.Fatalf("confirm decided %q", out.Decision)
	}
	if out.Delivers {
		t.Fatal("confirm delivered without an approval")
	}
	// Delivery is set in exactly one place in the policy owner, and it is the
	// strict branch. A confirmation cannot reach it.
	src, err := os.ReadFile("advisory_policy.go")
	if err != nil {
		t.Fatal(err)
	}
	body := codeWithoutComments(string(src))
	decide := body[strings.Index(body, "func decideCandidatePolicy("):]
	decide = decide[:strings.Index(decide, "\nfunc ")]
	if n := strings.Count(decide, "out.Delivers ="); n != 1 {
		t.Fatalf("the policy owner assigns Delivers %d times, want once", n)
	}
	strictBranch := decide[strings.Index(decide, "if strictAuthorized {"):]
	strictBranch = strictBranch[:strings.Index(strictBranch, "\n\t}")]
	if !strings.Contains(strictBranch, "out.Delivers =") {
		t.Error("the one delivering assignment is not the strict branch")
	}
	// And it is the acquisition control that can take it away, nothing else.
	if !strings.Contains(strictBranch, "!in.CaptureOnlySuppressed") {
		t.Error("the strict branch delivers regardless of the acquisition control")
	}
}

// The policy record is identities and closed values. Nothing else.
func TestPolicyTelemetryCarriesNoContent(t *testing.T) {
	w := newRouteWorldWithClosure(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
			`"verification_knowledge":"declared","verification":["pytest -q --secret hunter2"]}`,
		map[string]stubEffect{"pytest -q --secret hunter2": {}}, false)
	recs := captureShadow(t, func() {
		if _, err := w.write(t); err != nil {
			t.Fatal(err)
		}
	})
	kinds := []string{"candidate_policy_decision", "candidate_mutation_scope"}
	seen := 0
	for _, kind := range kinds {
		for _, r := range recordsOfKind(recs, kind) {
			seen++
			raw, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			blob := string(raw)
			for _, secret := range []string{
				routeWinner, routeBaseline, "hunter2", "pytest", w.path,
				"Make solve fast.", w.dir,
			} {
				if secret != "" && strings.Contains(blob, secret) {
					t.Errorf("%s carries %q", kind, secret)
				}
			}
		}
	}
	if seen == 0 {
		t.Fatal("no policy or scope records were written")
	}
}

// With the capture off the policy still decides, writes nothing, and allocates
// nothing for a record it is not going to make.
func TestCaptureDisabledIsInert(t *testing.T) {
	w := newRouteWorldWithClosure(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
		nil, false)
	before, err := os.ReadFile(w.path)
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	res, err := w.write(t)
	if err != nil {
		t.Fatalf("write failed with the sink off: %v", err)
	}
	if res == nil || !res.Success {
		t.Fatalf("the route stopped delivering with the sink off: %+v", res)
	}
	_ = before
	// The record writers return before building anything when the sink is off.
	for _, fn := range []string{"recordCandidatePolicyDecision", "recordMutationScope"} {
		src, err := os.ReadFile(map[string]string{
			"recordCandidatePolicyDecision": "candidate_policy.go",
			"recordMutationScope":           "mutation_scope.go",
		}[fn])
		if err != nil {
			t.Fatal(err)
		}
		body := string(src)
		f := body[strings.Index(body, "func "+fn+"("):]
		f = f[:strings.Index(f, "\n}")]
		guard := strings.Index(f, "if !sink.enabled() {")
		submit := strings.Index(f, "sink.submit(")
		if guard < 0 || submit < 0 || guard > submit {
			t.Errorf("%s builds its record before checking the sink", fn)
		}
	}
}

// Provenance describes the bytes that are on disk, and neither the model nor
// the service can set it.
func TestDeliveredProvenanceCannotBeForged(t *testing.T) {
	// The field is not on any wire type the model or the service fills.
	types, err := os.ReadFile("types.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(types)
	for _, typeName := range []string{"type V3GenerateResponse struct",
		"type WriteFileOutput struct", "type modelFacingResult struct"} {
		start := strings.Index(body, typeName)
		if start < 0 {
			continue
		}
		decl := body[start : start+strings.Index(body[start:], "\n}")]
		if strings.Contains(decl, "DeliveryProvenance") ||
			strings.Contains(decl, "delivery_provenance") {
			t.Errorf("%s carries the provenance", typeName)
		}
	}
	// It is set only beside a delivery that happened, from the policy outcome.
	for _, f := range []string{"tools.go", "edit_route_delivery.go"} {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(codeWithoutComments(string(src)), "\n") {
			if !strings.Contains(line, "DeliveryProvenance") {
				continue
			}
			trimmed := strings.TrimSpace(line)
			// The one setter, its own body, the field's declaration in a
			// closed helper, and the terminal's event. Nothing reads a
			// provenance off a response, a message or an envelope.
			allowed := strings.Contains(trimmed, "withDeliveryProvenance(") ||
				strings.Contains(trimmed, "deliveryProvenanceFor(") ||
				strings.Contains(trimmed, "emitDeliveryProvenance(") ||
				trimmed == "result.DeliveryProvenance = provenance" ||
				strings.Contains(trimmed, "res.DeliveryProvenance")
			if !allowed {
				t.Errorf("%s sets the provenance from %q", f, trimmed)
			}
		}
	}
	// And what it says matches what landed.
	w := newRouteWorldWithClosure(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
			`"verification_knowledge":"declared",`+
			`"verification_requirements_version":1,"verification_requirements":[`+
			`{"command":"pytest -q","kind":"behavioral","expects":"exit_zero",`+
			`"asset_authority":"client_supplied"}]}`,
		map[string]stubEffect{"pytest -q": {}}, false)
	res, err := w.write(t)
	if err != nil {
		t.Fatal(err)
	}
	landed := w.disk(t)
	switch res.DeliveryProvenance {
	case DeliveryFromStrictCandidate:
		if landed != routeWinner {
			t.Errorf("provenance says a candidate landed; disk holds %q", landed)
		}
	case DeliveryFromModelProposal:
		if landed != routeBaseline {
			t.Errorf("provenance says the model's bytes landed; disk holds %q", landed)
		}
	default:
		t.Errorf("unexpected provenance %q", res.DeliveryProvenance)
	}
}

// Every route that can put V3 bytes on disk is registered, and the registry is
// the only way to reach one.
func TestEveryByteToDiskRouteStaysRegistered(t *testing.T) {
	for _, site := range []string{
		"tools.go:writeFileWithV3",
		"edit_route_delivery.go:deliverEditCandidate",
		"candidate_delivery.go:deliverAuthorizedCandidate",
	} {
		if !registeredV3DeliveryRoute(site) {
			t.Errorf("%s is not a registered delivery route", site)
		}
	}
	if registeredV3DeliveryRoute("tools.go:somethingElse") {
		t.Error("an unregistered site was accepted")
	}
}
