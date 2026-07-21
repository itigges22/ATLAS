package main

// Structural gate for the edit and write paths (issue #147). The V3
// structural veto hard-rejects generated candidates whose direct-identifier
// calls resolve to no local def, import, or builtin — but the edit path
// (improveContentWithV3) frequently sent no project_context, so the
// in-pipeline veto was gated off, and even when it fired the pipeline's
// baseline fallback resurrected the model's own edit. Result observed in
// 2026-07-18 dogfooding: an ast_edit replaced a route with a body calling
// render_template while the file imported only render_template_string; it
// passed V3 verification, landed as verified, and every request 500'd
// (NameError). ast_edit had no syntax gate at all; edit_file's syntax gate
// catches parse failures but a NameError parses fine.
//
// This proxy-side gate closes the hole where it can't be bypassed: it
// resolves the COMPOSED post-change file through v3-service's structural
// checker and refuses landing content that INTRODUCES an unresolved direct
// call — the same healthy->broken rule as the syntax gate (a change that
// leaves a pre-existing unresolved name in place, i.e. a repair-in-
// progress, is allowed). Wired into edit_file, ast_edit, and every
// write_file branch (V3 winner, V3-error fallback, iteration fast-path,
// T0/T1 direct); under BypassV3 only the non-iterating T0/T1 direct
// write_file skips the gate (so the demo baseline pane shows the raw
// model) — the edit paths and the iteration fast-path stay gated in all
// modes. Python-only and fail-open: if v3-service is unreachable, the file
// isn't .py, or tree-sitter is unavailable, the write proceeds — the gate
// only blocks on a POSITIVE, newly-introduced unresolved call.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// checkStructuralUnresolved returns the direct-identifier calls in
// `content` that resolve to nothing (no local def, import, builtin, or
// supplied project symbol), and ok=true only when the check actually ran.
// Fail-open: (nil, false) for a non-.py file, an empty V3 URL, a network
// failure, or a tree-sitter/parse error on the far side.
func checkStructuralUnresolved(ctx *AgentContext, path, content string) ([]string, bool) {
	if ctx == nil || ctx.V3URL == "" {
		return nil, false
	}
	if strings.ToLower(filepath.Ext(path)) != ".py" {
		return nil, false
	}
	payload := map[string]interface{}{"path": path, "source": content}
	// Pass the OTHER files the model has read as project context so a call
	// to a symbol defined elsewhere in the project is credited (more
	// lenient = fewer false blocks). Crucially, EXCLUDE the file being
	// edited: SnapshotFilesRead still holds its PRE-EDIT body, which would
	// credit a top-level def the edit just deleted and let a genuine
	// NameError through (#147 review finding #2). The edited file's current
	// symbols come from `source`, which structural_score parses directly.
	cleanTarget := filepath.Clean(path)
	rel := make(map[string]string)
	addContext := func(p, c string) {
		if filepath.Clean(p) == cleanTarget {
			return // don't credit the pre-edit self
		}
		r, err := filepath.Rel(ctx.WorkingDir, p)
		if err != nil || r == "" {
			r = p
		}
		// Only .py files carry resolvable symbols, and entries are
		// truncated like the V3 request builders — read_file snapshots
		// can be 200 KB each, and this body is POSTed per gated write.
		if strings.ToLower(filepath.Ext(r)) != ".py" {
			return
		}
		if len(c) > 4000 {
			c = c[:4000] + "\n... (truncated)"
		}
		rel[r] = c
	}
	for p, c := range ctx.SnapshotFilesRead() {
		addContext(p, c)
	}
	// Files the session WROTE are leniency context too — write_file paths
	// never RecordFileRead, so without these a sibling the session just
	// created is invisible here while the in-pipeline veto (which merges
	// SessionWrites) credits it, making this gate strictly stricter than
	// the veto it backstops. Disk content wins over any stale snapshot.
	for w := range ctx.SessionWrites {
		if w == "" || strings.ToLower(filepath.Ext(w)) != ".py" {
			continue // only .py carries symbols — skip the disk read otherwise
		}
		abs := resolveAgentPath(ctx, w)
		if filepath.Clean(abs) == cleanTarget {
			continue // pre-edit self; skip the read
		}
		if data, err := os.ReadFile(abs); err == nil {
			addContext(abs, string(data))
		}
	}
	if len(rel) > 0 {
		payload["project_context"] = rel
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, false
	}
	// ctx.Ctx may be nil on paths constructed without a request context;
	// the gate must fail open (or keep working), never panic.
	base := ctx.Ctx
	if base == nil {
		base = context.Background()
	}
	reqCtx, cancel := context.WithTimeout(base, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "POST",
		ctx.V3URL+"/internal/structural_check", bytes.NewReader(body))
	if err != nil {
		return nil, false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, false // fail-open
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, false
	}
	var r struct {
		OK         bool     `json:"ok"`
		Unresolved []string `json:"unresolved"`
	}
	if json.Unmarshal(raw, &r) != nil || !r.OK {
		return nil, false // parse error / tree-sitter missing -> fail-open
	}
	return r.Unresolved, true
}

// editIntroducesUnresolved returns the names an edit NEWLY makes
// unresolved — present in the edited file's unresolved set but not the
// original's. Mirrors the syntax gate's healthy->broken rule: an edit must
// not INTRODUCE a NameError, but a pre-existing unresolved name (the model
// is mid-repair) is allowed to remain. Returns nil when the check couldn't
// run (fail-open) or nothing new was introduced.
func editIntroducesUnresolved(ctx *AgentContext, path, original, edited string) []string {
	editedUnres, ok := checkStructuralUnresolved(ctx, path, edited)
	if !ok || len(editedUnres) == 0 {
		return nil
	}
	origUnres, ok := checkStructuralUnresolved(ctx, path, original)
	if !ok {
		// One retry: the edited-side call just succeeded, so a failure
		// here is a transient blip on the second back-to-back request.
		origUnres, ok = checkStructuralUnresolved(ctx, path, original)
	}
	if !ok {
		// The original-side check couldn't run (transient service failure;
		// tree-sitter-missing would have failed the edited side first, and
		// malformed Python does NOT trigger this — tree-sitter parses it
		// tolerantly and returns a partial extraction). Without a baseline
		// the healthy->broken comparison is meaningless, and counting
		// EVERY unresolved name as newly introduced would block the model
		// from fixing one error at a time — fail open instead.
		return nil
	}
	was := make(map[string]bool, len(origUnres))
	for _, n := range origUnres {
		was[n] = true
	}
	var introduced []string
	for _, n := range editedUnres {
		if !was[n] {
			introduced = append(introduced, n)
		}
	}
	return introduced
}

// readOriginalForGate returns the on-disk original for the healthy->broken
// comparison. A missing file is a first write (empty original — every
// unresolved call counts as introduced). Any OTHER read failure means the
// original is unknowable, so the caller must skip the gate (fail open)
// rather than treat the file as empty and count pre-existing unresolved
// calls as newly introduced.
func readOriginalForGate(path string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", true
		}
		return "", false
	}
	return string(data), true
}

// structuralRejection builds the tool error handed back when the gate
// blocks an edit — names the offending calls and the recovery.
// structuralWriteRejection is the write_file variant: the recovery must
// name the operation the model actually issued — an "edit" steer on a
// blocked NEW-file write sends the model to edit_file against a file
// that doesn't exist.
func structuralRejection(path string, introduced []string) string {
	return fmt.Sprintf(
		"edit for %s calls %s, which the file neither imports, defines, nor "+
			"gets from builtins — running it would raise NameError. The file was "+
			"NOT modified. Add the missing import (or correct the name to one that "+
			"IS in scope), then re-issue the edit.",
		path, quoteNames(introduced))
}

func structuralWriteRejection(path string, introduced []string) string {
	return fmt.Sprintf(
		"write_file for %s calls %s, which the file neither imports, defines, "+
			"nor gets from builtins — running it would raise NameError. Nothing "+
			"was written. Add the missing import (or correct the name to one that "+
			"IS in scope), then re-issue the write_file with the full corrected "+
			"content.",
		path, quoteNames(introduced))
}

func quoteNames(names []string) string {
	quoted := make([]string, len(names))
	for i, n := range names {
		quoted[i] = "`" + n + "`"
	}
	return strings.Join(quoted, ", ")
}
