package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// resolveWorkspacePath translates an approved host path and verifies that the
// result stays inside the configured workspace. It checks both the lexical path
// and the nearest existing ancestor so symlinks cannot escape the workspace.
func resolveWorkspacePath(ctx *AgentContext, path string) (string, error) {
	resolved := resolveAgentPath(ctx, path)
	root, err := filepath.Abs(ctx.WorkingDir)
	if err != nil {
		return "", fmt.Errorf("resolve workspace root: %w", err)
	}
	candidate, err := filepath.Abs(resolved)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(candidate))
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q is outside the workspace", path)
	}

	workspace, err := os.OpenRoot(root)
	if err != nil {
		return "", fmt.Errorf("open workspace root: %w", err)
	}
	defer workspace.Close()
	existing := rel
	for {
		if _, statErr := workspace.Stat(existing); statErr == nil {
			break
		} else if !os.IsNotExist(statErr) {
			return "", fmt.Errorf("inspect path %q: %w", path, statErr)
		}
		parent := filepath.Dir(existing)
		if parent == existing || existing == "." {
			return "", fmt.Errorf("path %q has no existing workspace ancestor", path)
		}
		existing = parent
	}
	return candidate, nil
}

// readWorkspaceFile opens a file through os.Root, so the read remains confined
// even if a workspace symlink is swapped between validation and use.
func readWorkspaceFile(ctx *AgentContext, path string) ([]byte, string, error) {
	resolved, err := resolveWorkspacePath(ctx, path)
	if err != nil {
		return nil, "", err
	}
	rootPath, err := filepath.Abs(ctx.WorkingDir)
	if err != nil {
		return nil, "", fmt.Errorf("resolve workspace root: %w", err)
	}
	rel, err := filepath.Rel(rootPath, resolved)
	if err != nil {
		return nil, "", fmt.Errorf("resolve workspace-relative path: %w", err)
	}
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		return nil, "", fmt.Errorf("open workspace root: %w", err)
	}
	defer root.Close()
	file, err := root.Open(rel)
	if err != nil {
		return nil, "", err
	}
	data, readErr := io.ReadAll(file)
	closeErr := file.Close()
	if readErr != nil {
		return nil, "", readErr
	}
	if closeErr != nil {
		return nil, "", closeErr
	}
	return data, resolved, nil
}

// readWorkspaceDir is the directory equivalent of readWorkspaceFile.
func readWorkspaceDir(ctx *AgentContext, path string) ([]os.DirEntry, error) {
	resolved, err := resolveWorkspacePath(ctx, path)
	if err != nil {
		return nil, err
	}
	rootPath, err := filepath.Abs(ctx.WorkingDir)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace root: %w", err)
	}
	rel, err := filepath.Rel(rootPath, resolved)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace-relative path: %w", err)
	}
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		return nil, fmt.Errorf("open workspace root: %w", err)
	}
	defer root.Close()
	dir, err := root.Open(rel)
	if err != nil {
		return nil, err
	}
	defer dir.Close()
	return dir.ReadDir(-1)
}

// validateToolWorkspacePaths applies workspace containment before any tool
// handler can touch the filesystem. It is used by both the agent loop and the
// shared dispatcher so parallel and direct dispatch paths follow one policy.
func validateToolWorkspacePaths(name string, args json.RawMessage, ctx *AgentContext) string {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(args, &fields); err != nil {
		return ""
	}
	keys := map[string][]string{
		"read_file":      {"path"},
		"outline_file":   {"path"},
		"write_file":     {"path"},
		"edit_file":      {"path"},
		"ast_edit":       {"path"},
		"delete_file":    {"path"},
		"move_file":      {"source", "destination"},
		"search_files":   {"path"},
		"find_file":      {"path"},
		"list_directory": {"path"},
		"run_command":    {"cwd"},
		"run_background": {"cwd"},
	}
	for _, key := range keys[name] {
		raw, ok := fields[key]
		if !ok {
			continue
		}
		var value string
		if json.Unmarshal(raw, &value) != nil || strings.TrimSpace(value) == "" {
			continue
		}
		if _, err := resolveWorkspacePath(ctx, value); err != nil {
			return fmt.Sprintf("%s: %v", name, err)
		}
	}
	return ""
}
