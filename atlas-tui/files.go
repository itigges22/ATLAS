// PC-062 follow-up: file-tree sidebar.
//
// Renders the workspace root as a left sidebar so devs can see what's
// in the project at a glance without bouncing to a terminal pane. Two
// levels deep, sorted (dirs first, then files alphabetically), with
// recently-modified files highlighted (the agent just wrote them).

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// fileEntry is one row in the sidebar tree.
type fileEntry struct {
	RelPath string // path relative to scan root, e.g. "src/main.go"
	IsDir   bool
	Depth   int
}

// scanIgnored returns true for directory names we never recurse into.
// Mostly noise that bloats the tree (build artifacts, VCS internals,
// language caches). Keeping this list small and explicit — better to
// show a few extra dirs than hide something the user cared about.
var scanIgnored = map[string]bool{
	".git":          true,
	"node_modules":  true,
	"__pycache__":   true,
	".venv":         true,
	"venv":          true,
	"dist":          true,
	"build":         true,
	".idea":         true,
	".vscode":       true,
	"target":        true,
	".next":         true,
	".nuxt":         true,
	".cache":        true,
	".pytest_cache": true,
	".mypy_cache":   true,
	".ruff_cache":   true,
	"__MACOSX":      true,
}

// scanFiles walks `root` to `maxDepth` directory levels and returns a
// flattened, ordered list of entries suitable for sidebar rendering.
// Caps at `maxEntries` total to keep huge monorepos from blowing up.
func scanFiles(root string, maxDepth, maxEntries int) []fileEntry {
	if root == "" {
		return nil
	}
	out := make([]fileEntry, 0, 64)
	var walk func(dir string, depth int)
	walk = func(dir string, depth int) {
		if len(out) >= maxEntries {
			return
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		// Sort so dirs come before files at each level, alphabetical
		// within each group. Stable per-tick scans keep the sidebar
		// from jittering as files are added/removed.
		sort.SliceStable(entries, func(i, j int) bool {
			if entries[i].IsDir() != entries[j].IsDir() {
				return entries[i].IsDir()
			}
			return strings.ToLower(entries[i].Name()) <
				strings.ToLower(entries[j].Name())
		})
		for _, e := range entries {
			if len(out) >= maxEntries {
				return
			}
			name := e.Name()
			if e.IsDir() && scanIgnored[name] {
				continue
			}
			full := filepath.Join(dir, name)
			rel, err := filepath.Rel(root, full)
			if err != nil {
				continue
			}
			out = append(out, fileEntry{
				RelPath: rel,
				IsDir:   e.IsDir(),
				Depth:   depth,
			})
			if e.IsDir() && depth < maxDepth {
				walk(full, depth+1)
			}
		}
	}
	walk(root, 0)
	return out
}

var (
	fileDirStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("117")).
			Bold(true)

	fileFileStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("251"))

	fileModifiedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("214")).
				Bold(true)
)

// renderFilesPane returns the sidebar contents (no border — caller
// wraps in a bordered box at the right size). `modified` is the set
// of relative paths that were touched by recent tool calls; those rows
// render in orange/bold so the user can see what just changed.
//
// Returns the rendered string plus the total line count, for caller
// to optionally show a scroll indicator (future).
func renderFilesPane(entries []fileEntry, modified map[string]bool,
	root string, height, width, scroll int) string {
	if height <= 0 {
		return ""
	}
	if len(entries) == 0 {
		return dimStyle.Render("(empty workspace)")
	}

	// Header row showing the scan root (basename only — we don't have
	// width to spare). Helps the user confirm "yes, this is where the
	// agent is operating".
	rootDisplay := filepath.Base(root)
	if rootDisplay == "" || rootDisplay == "/" {
		rootDisplay = root
	}
	header := dimStyle.Render(truncate("● "+rootDisplay, width))

	rows := []string{header}
	for _, e := range entries {
		indent := strings.Repeat("  ", e.Depth)
		name := filepath.Base(e.RelPath)
		var marker string
		var styled string
		switch {
		case e.IsDir:
			marker = "▸ "
			styled = fileDirStyle.Render(indent + marker + name + "/")
		case modified[e.RelPath]:
			marker = "● "
			styled = fileModifiedStyle.Render(indent + marker + name)
		default:
			marker = "  "
			styled = fileFileStyle.Render(indent + marker + name)
		}
		// Truncate the rendered string by visible width — lipgloss
		// strings include ANSI codes that don't count toward width.
		// Quick approximation: truncate raw, re-render with same style.
		raw := indent + marker + name
		if e.IsDir {
			raw += "/"
		}
		if lipgloss.Width(styled) > width {
			cut := truncate(raw, width)
			switch {
			case e.IsDir:
				styled = fileDirStyle.Render(cut)
			case modified[e.RelPath]:
				styled = fileModifiedStyle.Render(cut)
			default:
				styled = fileFileStyle.Render(cut)
			}
		}
		rows = append(rows, styled)
	}

	// Scroll: clamp `scroll` against total length, take a window.
	total := len(rows)
	maxScroll := total - height
	if maxScroll < 0 {
		maxScroll = 0
	}
	if scroll > maxScroll {
		scroll = maxScroll
	}
	if scroll < 0 {
		scroll = 0
	}
	end := total - scroll
	if end > total {
		end = total
	}
	start := end - height
	if start < 0 {
		start = 0
	}
	out := append([]string(nil), rows[start:end]...)
	for len(out) < height {
		out = append(out, "")
	}
	if total > height {
		// Quiet hint at the bottom that there's more.
		more := fmt.Sprintf("(+%d more)", total-height-scroll)
		if scroll > 0 {
			more = fmt.Sprintf("(↑%d / +%d more)", scroll, total-height-scroll)
		}
		if len(out) > 0 {
			out[len(out)-1] = dimStyle.Render(truncate(more, width))
		}
	}
	return strings.Join(out, "\n")
}

// extractWritePath pulls the file path out of a tool call's args JSON
// for tools that touch a single file. Used to highlight modified files
// in the sidebar. Returns "" if the args don't fit the pattern.
func extractWritePath(args []byte) string {
	// We only need the "path" field — keep it cheap.
	var p struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(args, &p); err != nil {
		return ""
	}
	return p.Path
}
