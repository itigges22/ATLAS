package main

// Session file manifest. The agent creates multi-file projects one
// write at a time, and each write is generated as if it were the first
// file in an empty project — SessionWrites knows what exists, but that
// knowledge never reached the model or V3's candidate generation.
// Observed 2026-07-18: the model wrote templates/index.html and
// static/game.js, then generated an app.py that inlined its own page
// with render_template_string, orphaning both of its earlier files.
//
// Two consumers:
//   - sessionManifestNote: a [system note] appended to the conversation
//     when the session's file set grows past one file, listing what
//     exists so later files reference earlier ones.
//   - mergeSessionWritesIntoContext (tools.go): puts session-written
//     files into V3GenerateRequest.ProjectContext, which previously
//     held only files the model had READ — written-but-never-read
//     files were invisible to candidate generation.

import (
	"fmt"
	"sort"
	"strings"
)

// sessionManifestNote returns a one-shot context note listing the files
// this session has created, or "" when there is nothing new to say.
// Fires only when the set grew AND holds at least two files — a single
// file has no cross-file relationships worth announcing. Announced
// state lives in ctx.ManifestAnnounced so each file is announced once.
func sessionManifestNote(ctx *AgentContext) string {
	if len(ctx.SessionWrites) < 2 {
		return ""
	}
	if ctx.ManifestAnnounced == nil {
		ctx.ManifestAnnounced = make(map[string]bool)
	}
	fresh := false
	paths := make([]string, 0, len(ctx.SessionWrites))
	for p := range ctx.SessionWrites {
		paths = append(paths, p)
		if !ctx.ManifestAnnounced[p] {
			ctx.ManifestAnnounced[p] = true
			fresh = true
		}
	}
	if !fresh {
		return ""
	}
	sort.Strings(paths)
	return fmt.Sprintf(
		"Project files created this session: %s. When these should work "+
			"together, reference them (e.g. render_template(\"page.html\"), "+
			"<script src=\"/static/app.js\">) instead of re-creating or "+
			"inlining their content. Files you have not re-read may have "+
			"been improved by build verification since you wrote them.",
		strings.Join(paths, ", "))
}
