package main

import (
	"go/ast"
	"strings"
	"testing"
)

// The delivery graph, enumerated.
//
// Two producers now exist and each computes authority. Nothing consults it.
// This file is the proof: every production consumer of provenance is named
// here, and a new one fails the build's tests rather than quietly shipping.
//
// The rule is not "provenance is unused" -- it is computed, serialised and
// compared. The rule is that no result of it reaches a decision about what
// lands on disk.

// deliveryDecisions are the functions that decide whether a generated
// candidate may replace the caller's content. Adding a caller to any of them
// changes what lands, so the set of callers is pinned.
var deliveryDecisions = []string{
	"EvidenceSupportsProvenanceFor",
	"authorizedV3Replacement",
	"v3DeliveryAuthorized",
}

// provenanceReaders are the ways a piece of provenance can be turned into a
// yes or a no. A production caller of one of these is a consumer of trust.
var provenanceReaders = []string{
	"MayAuthorize", "BindsTo", "Authorizes",
	"produceSyntaxEvidence", "produceDeclaredVerificationEvidence",
	"declaredVerificationCoverage",
}

// TestEveryProductionConsumerOfProvenanceIsEnumerated fails when a call to a
// provenance reader appears anywhere in production code.
//
// The producers are the exception, and only for their own construction: a
// producer may build a binding, and Authorizes may be called by the coverage
// helper that reports what is still owed. Neither result reaches a write.
func TestEveryProductionConsumerOfProvenanceIsEnumerated(t *testing.T) {
	allowed := map[string]bool{
		// The coverage helper reports which declared commands are still owed.
		// Its answer is returned to its caller and, in this build, has none.
		"verification_evidence.go:declaredVerificationCoverage": true,
	}
	for _, fn := range provenanceReaders {
		for site := range callSites(proxyFiles(t), fn) {
			if allowed[site] {
				continue
			}
			t.Errorf("%s calls %s: every production consumer of provenance must "+
				"be enumerated here, and this one is not", site, fn)
		}
	}
}

// TestTheDeliveryGraphGainedNoCaller pins the callers of the three functions
// that decide what lands. The set is exactly what it was at dc9172c.
func TestTheDeliveryGraphGainedNoCaller(t *testing.T) {
	allowed := map[string]bool{
		"v3_bridge.go:v3DeliveryAuthorized":          true,
		"v3_bridge.go:EvidenceSupportsProvenanceFor": true,
		"tools.go:authorizedV3Replacement":           true,
		"tools.go:writeFileWithV3":                   true,
		"tools.go:improveContentWithV3":              true,
	}
	for _, fn := range deliveryDecisions {
		for site := range callSites(proxyFiles(t), fn) {
			if !allowed[site] {
				t.Errorf("%s gained a new caller of %s", site, fn)
			}
		}
	}
}

// TestNoProducerReachesTheDeliveryGraph pins the direction: a producer may
// never call a delivery decision, and a delivery decision may never call a
// producer.
func TestNoProducerReachesTheDeliveryGraph(t *testing.T) {
	files := proxyFiles(t)
	producers := map[string]bool{
		"produceSyntaxEvidence":               true,
		"produceDeclaredVerificationEvidence": true,
		"declaredVerificationCoverage":        true,
		"deriveTaskObligations":               true,
		"newTaskObligation":                   true,
	}
	for _, fn := range deliveryDecisions {
		for site := range callSites(files, fn) {
			caller := site[strings.Index(site, ":")+1:]
			if producers[caller] {
				t.Errorf("producer %s calls the delivery decision %s", caller, fn)
			}
		}
	}
	for fn := range producers {
		for site := range callSites(files, fn) {
			for _, d := range deliveryDecisions {
				if strings.HasSuffix(site, ":"+d) {
					t.Errorf("delivery decision %s calls the producer %s", d, fn)
				}
			}
		}
	}
}

// TestTheWritePathReadsNoObligationOrEvidence pins that the functions which
// actually put bytes on disk consult none of this slice's vocabulary.
func TestTheWritePathReadsNoObligationOrEvidence(t *testing.T) {
	files := proxyFiles(t)
	writePath := map[string]bool{
		"writeFileWithV3":               true,
		"improveContentWithV3":          true,
		"authorizedV3Replacement":       true,
		"v3DeliveryAuthorized":          true,
		"EvidenceSupportsProvenanceFor": true,
	}
	banned := append(append([]string{}, provenanceReaders...),
		"deriveTaskObligations", "newTaskObligation", "obligationID",
		"obligationClosureFloor", "baselineEvidenceStrength",
		"baselineIdentityFor", "workspaceIdentity")
	for _, fn := range banned {
		for site := range callSites(files, fn) {
			caller := site[strings.Index(site, ":")+1:]
			if writePath[caller] {
				t.Errorf("the write path function %s calls %s", caller, fn)
			}
		}
	}
}

// TestTheProducersAreCalledByNothingInProduction is the strongest statement
// this build can make: the two producers exist, are tested, and have no
// production caller at all.
func TestTheProducersAreCalledByNothingInProduction(t *testing.T) {
	files := proxyFiles(t)
	for _, fn := range []string{"produceSyntaxEvidence", "produceDeclaredVerificationEvidence"} {
		sites := callSites(files, fn)
		if len(sites) != 0 {
			t.Errorf("%s has production callers %v; activating a producer changes "+
				"what lands and is a separate slice", fn, sites)
		}
	}
}

// TestTerminalCompletionReadsNoEvidence pins that the terminal decision is
// untouched: it consults the obligation DECISION that landed at dc9172c and
// none of this slice's evidence vocabulary.
func TestTerminalCompletionReadsNoEvidence(t *testing.T) {
	files := proxyFiles(t)
	terminal := map[string]bool{
		"finalizeCompletion":        true,
		"terminalCompletionAllowed": true,
		"decideVerificationDemand":  true,
		"missingExpectedOutputs":    true,
	}
	banned := append(append([]string{}, provenanceReaders...),
		"deriveTaskObligations", "obligationID", "obligationClosureFloor",
		"baselineIdentityFor", "workspaceIdentity")
	for _, fn := range banned {
		for site := range callSites(files, fn) {
			caller := site[strings.Index(site, ":")+1:]
			if terminal[caller] {
				t.Errorf("the terminal decision %s calls %s", caller, fn)
			}
		}
	}
}

// TestNoProductionCodeConstructsAProvenanceOutsideTheProducers pins that a
// binding can only be built where the rules for building one live.
func TestNoProductionCodeConstructsAProvenanceOutsideTheProducers(t *testing.T) {
	allowed := map[string]bool{
		"syntax_evidence.go":       true,
		"verification_evidence.go": true,
		// The type and its methods live here.
		"types.go": true,
		// The wire decoder materialises what the service sent.
		"v3_bridge.go": true,
	}
	for name, f := range proxyFiles(t) {
		if allowed[name] {
			continue
		}
		ast.Inspect(f, func(n ast.Node) bool {
			lit, ok := n.(*ast.CompositeLit)
			if !ok {
				return true
			}
			if id, ok := lit.Type.(*ast.Ident); ok && id.Name == "V3EvidenceProvenance" {
				t.Errorf("%s constructs a V3EvidenceProvenance outside the producers", name)
			}
			return true
		})
	}
}
