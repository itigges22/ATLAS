package main

// Structural gate for the edit path (issue #147). The V3 structural veto
// hard-rejects generated candidates whose direct-identifier calls resolve
// to no local def, import, or builtin — but the edit path
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
// resolves the COMPOSED post-edit file through v3-service's structural
// checker and refuses a write that INTRODUCES an unresolved direct call —
// the same healthy->broken rule as the syntax gate (an edit that leaves a
// pre-existing unresolved name in place, i.e. a repair-in-progress, is
// allowed). Python-only and fail-open: if v3-service is unreachable, the
// file isn't .py, or tree-sitter is unavailable, the write proceeds — the
// gate only blocks on a POSITIVE, newly-introduced unresolved call.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
	// Pass the files the model has read as project context so a call to a
	// symbol defined elsewhere in the project is credited (more lenient =
	// fewer false blocks); the render_template class is caught by the
	// file's OWN imports regardless.
	if pc := ctx.SnapshotFilesRead(); len(pc) > 0 {
		rel := make(map[string]string, len(pc))
		for p, c := range pc {
			r, err := filepath.Rel(ctx.WorkingDir, p)
			if err != nil || r == "" {
				r = p
			}
			rel[r] = c
		}
		payload["project_context"] = rel
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, false
	}
	reqCtx, cancel := context.WithTimeout(ctx.Ctx, 5*time.Second)
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
	origUnres, _ := checkStructuralUnresolved(ctx, path, original)
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

// structuralRejection builds the tool error handed back when the gate
// blocks an edit — names the offending calls and the recovery.
func structuralRejection(path string, introduced []string) string {
	quoted := make([]string, len(introduced))
	for i, n := range introduced {
		quoted[i] = "`" + n + "`"
	}
	return fmt.Sprintf(
		"edit for %s calls %s, which the file neither imports, defines, nor "+
			"gets from builtins — running it would raise NameError. The file was "+
			"NOT modified. Add the missing import (or correct the name to one that "+
			"IS in scope), then re-issue the edit.",
		path, strings.Join(quoted, ", "))
}
