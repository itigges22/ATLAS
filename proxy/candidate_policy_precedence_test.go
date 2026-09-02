package main

import (
	"testing"
)

// Precedence, pinned: a client's explicit policy is applied over the operator
// default in both directions, an omitted policy inherits the operator default
// (the legacy and API contract, unchanged), and an unreadable operator value
// falls back to strict. What a client displays must be what it sends; this is
// the proxy's half of that agreement.
func TestCandidatePolicyPrecedenceMatrix(t *testing.T) {
	cases := []struct {
		operator string
		contract string
		want     candidatePolicyMode
		source   candidatePolicySource
	}{
		// operator says automatic
		{"automatic_v3", `{"task_mode":"work"}`, CandidatePolicyAutomaticV3, CandidatePolicySourceOperator},
		{"automatic_v3", `{"task_mode":"work","candidate_policy":"strict"}`, CandidatePolicyStrict, CandidatePolicySourceClient},
		{"automatic_v3", `{"task_mode":"work","candidate_policy":"advisory"}`, CandidatePolicyAdvisory, CandidatePolicySourceClient},
		{"automatic_v3", `{"task_mode":"work","candidate_policy":"automatic_v3"}`, CandidatePolicyAutomaticV3, CandidatePolicySourceClient},
		// operator says strict (or nothing)
		{"strict", `{"task_mode":"work"}`, CandidatePolicyStrict, CandidatePolicySourceDefault},
		{"", `{"task_mode":"work"}`, CandidatePolicyStrict, CandidatePolicySourceDefault},
		{"", `{"task_mode":"work","candidate_policy":"strict"}`, CandidatePolicyStrict, CandidatePolicySourceClient},
		{"", `{"task_mode":"work","candidate_policy":"advisory"}`, CandidatePolicyAdvisory, CandidatePolicySourceClient},
		{"", `{"task_mode":"work","candidate_policy":"automatic_v3"}`, CandidatePolicyAutomaticV3, CandidatePolicySourceClient},
		// operator says advisory; explicit strict still wins
		{"advisory", `{"task_mode":"work","candidate_policy":"strict"}`, CandidatePolicyStrict, CandidatePolicySourceClient},
		{"advisory", `{"task_mode":"work"}`, CandidatePolicyAdvisory, CandidatePolicySourceOperator},
		// an unreadable operator value is the default, not a refusal
		{"confirm", `{"task_mode":"work"}`, CandidatePolicyStrict, CandidatePolicySourceDefault},
	}
	for _, c := range cases {
		t.Run(c.operator+"|"+c.contract, func(t *testing.T) {
			t.Setenv("ATLAS_CANDIDATE_POLICY", c.operator)
			dir := t.TempDir()
			ctx := &AgentContext{WorkingDir: dir}
			ctx.TaskContract = mustContract(t, dir, c.contract)
			mode, source := candidatePolicyOf(ctx)
			if mode != c.want || source != c.source {
				t.Errorf("operator=%q contract=%s -> %s from %s, want %s from %s",
					c.operator, c.contract, mode, source, c.want, c.source)
			}
		})
	}
	// A contractless request under an operator default: the operator's.
	t.Setenv("ATLAS_CANDIDATE_POLICY", "automatic_v3")
	if mode, source := candidatePolicyOf(&AgentContext{}); mode != CandidatePolicyAutomaticV3 || source != CandidatePolicySourceOperator {
		t.Errorf("contractless under operator automatic -> %s from %s", mode, source)
	}
}

// The same agreement at the delivery owner, through the real route: the
// applied policy is the request's explicit one, whatever the operator set.
func TestExplicitClientPolicyGovernsDeliveryOverTheOperatorDefault(t *testing.T) {
	cases := []struct {
		name      string
		operator  string
		contract  string
		delivered bool
	}{
		{"operator automatic, client explicit strict", "automatic_v3", `{"task_mode":"work","candidate_policy":"strict"}`, false},
		{"operator automatic, client explicit advisory", "automatic_v3", `{"task_mode":"work","candidate_policy":"advisory"}`, false},
		{"operator automatic, client omitted (legacy inherits)", "automatic_v3", `{"task_mode":"work"}`, true},
		{"operator strict, client automatic", "strict", `{"task_mode":"work","candidate_policy":"automatic_v3"}`, true},
		{"operator strict, client advisory", "strict", `{"task_mode":"work","candidate_policy":"advisory"}`, false},
		{"operator strict, client omitted", "strict", `{"task_mode":"work"}`, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv("ATLAS_CANDIDATE_POLICY", c.operator)
			w := newAutomaticWorld(t, c.contract, routeWinner, nil, true)
			res, err := w.write(t)
			if err != nil || res == nil || !res.Success {
				t.Fatalf("write failed: %v %+v", err, res)
			}
			got := w.disk(t)
			if c.delivered && got != routeWinner {
				t.Fatalf("expected the selected candidate, disk holds the baseline")
			}
			if !c.delivered && got != routeBaseline {
				t.Fatalf("expected the model's own bytes, disk holds %q", got)
			}
			if (res.AuthorizedDeliveryHash != "") != c.delivered {
				t.Errorf("authorization spent=%v, want %v", res.AuthorizedDeliveryHash != "", c.delivered)
			}
		})
	}
}
