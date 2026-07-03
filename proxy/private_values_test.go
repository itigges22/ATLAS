// Corpus-driven tests for private-value filtering. Every fixture value
// is synthetic (see tests/fixtures/private_value_fixtures.json).

package main

import (
	"bytes"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type pvCase struct {
	Name           string   `json:"name"`
	Input          string   `json:"input"`
	MustNotContain []string `json:"must_not_contain"`
	MustContain    []string `json:"must_contain"`
	MustEqualInput bool     `json:"must_equal_input"`
}

type pvCorpus struct {
	Placeholder   string   `json:"placeholder"`
	Cases         []pvCase `json:"cases"`
	NegativeCases []pvCase `json:"negative_cases"`
}

func loadCorpus(t *testing.T) pvCorpus {
	t.Helper()
	path := filepath.Join("..", "tests", "fixtures",
		"private_value_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("fixture corpus missing: %v", err)
	}
	var c pvCorpus
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatalf("fixture corpus unparsable: %v", err)
	}
	return c
}

func TestFilterPrivateValuesCorpus(t *testing.T) {
	corpus := loadCorpus(t)
	if corpus.Placeholder != privateValuePlaceholder {
		t.Fatalf("placeholder drift: corpus %q vs code %q",
			corpus.Placeholder, privateValuePlaceholder)
	}
	for _, c := range corpus.Cases {
		got := filterPrivateValues(c.Input)
		for _, bad := range c.MustNotContain {
			if strings.Contains(got, bad) {
				t.Errorf("%s: %q survived filtering: %q", c.Name, bad, got)
			}
		}
		for _, keep := range c.MustContain {
			if !strings.Contains(got, keep) {
				t.Errorf("%s: context %q lost: %q", c.Name, keep, got)
			}
		}
		if len(c.MustNotContain) > 0 &&
			!strings.Contains(got, privateValuePlaceholder) {
			t.Errorf("%s: no placeholder in output: %q", c.Name, got)
		}
	}
	for _, c := range corpus.NegativeCases {
		if got := filterPrivateValues(c.Input); got != c.Input {
			t.Errorf("%s: benign input modified: %q -> %q",
				c.Name, c.Input, got)
		}
	}
}

func TestFilteringWriterOnLogger(t *testing.T) {
	var buf bytes.Buffer
	lg := log.New(filteringWriter{w: &buf}, "", 0)
	lg.Printf("turn failed: EXAMPLE_API_TOKEN=not-a-real-token status=500")
	out := buf.String()
	if strings.Contains(out, "not-a-real-token") {
		t.Fatalf("fixture value reached the log sink: %q", out)
	}
	if !strings.Contains(out, "status=500") {
		t.Fatalf("benign context lost: %q", out)
	}
	if !strings.Contains(out, privateValuePlaceholder) {
		t.Fatalf("placeholder missing: %q", out)
	}
}
