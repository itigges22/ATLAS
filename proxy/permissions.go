package main

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

// ---------------------------------------------------------------------------
// Permission system — controls which tool calls require user confirmation
// ---------------------------------------------------------------------------

// extractMatchValue extracts the path/command value from a tool call's
// args for the built-in safety deny-list below.
func extractMatchValue(toolName string, args json.RawMessage) string {
	switch toolName {
	case "run_command":
		var input RunCommandInput
		if err := json.Unmarshal(args, &input); err == nil {
			return input.Command
		}
	case "write_file":
		var input WriteFileInput
		if err := json.Unmarshal(args, &input); err == nil {
			return input.Path
		}
	case "edit_file":
		var input EditFileInput
		if err := json.Unmarshal(args, &input); err == nil {
			return input.Path
		}
	case "ast_edit":
		var input AstEditInput
		if err := json.Unmarshal(args, &input); err == nil {
			return input.Path
		}
	}
	return ""
}

// shellSegmentSplitter marks the boundaries between commands in a shell line
// (operators and grouping/substitution punctuation) so each segment's leading
// word can be inspected. Redirect targets (> file) are not command positions,
// so `>`/`<` are deliberately excluded.
var shellSegmentSplitter = strings.NewReplacer(
	";", "\n", "|", "\n", "&", "\n", "(", "\n", ")", "\n", "`", "\n",
)

// commandPrefixWords are leading words that wrap the real command word.
var commandPrefixWords = map[string]bool{
	"sudo": true, "doas": true, "env": true, "command": true, "nice": true,
	"nohup": true, "time": true, "exec": true, "builtin": true,
}

// denyCommandReason reports why a shell command is blocked, or "" if allowed.
// Matching is anchored to the command position of each shell segment so only
// the destructive form is blocked — `rm -rf /` but not `rm -rf /workspace`,
// `mkfs.ext4 /dev/sda` but not `grep mkfs notes.txt`, `dd of=/dev/sda` but not
// `dd of=out.bin`.
func denyCommandReason(cmd string) string {
	for _, seg := range strings.Split(shellSegmentSplitter.Replace(cmd), "\n") {
		fields := strings.Fields(seg)
		i := 0
		for i < len(fields) && commandPrefixWords[fields[i]] {
			i++
		}
		if i >= len(fields) {
			continue
		}
		head := fields[i]
		rest := fields[i+1:]
		switch {
		case head == "rm" && rmTargetsRoot(rest):
			return "blocked by safety rule: recursive removal of /"
		case head == "mkfs" || strings.HasPrefix(head, "mkfs."):
			return "blocked by safety rule: mkfs"
		case head == "dd":
			for _, a := range rest {
				if strings.HasPrefix(a, "of=/dev/") {
					return "blocked by safety rule: dd to a device"
				}
			}
		}
	}
	return ""
}

// rmTargetsRoot reports whether an `rm` argument list recursively targets the
// filesystem root (`/` or `/*`).
func rmTargetsRoot(args []string) bool {
	recursive, rootTarget := false, false
	for _, a := range args {
		if strings.HasPrefix(a, "-") {
			if a == "--recursive" || (!strings.HasPrefix(a, "--") && strings.ContainsAny(a, "rR")) {
				recursive = true
			}
			continue
		}
		if a == "/" || a == "/*" {
			rootTarget = true
		}
	}
	return recursive && rootTarget
}

// denyWritePathReason reports why writing to a path is blocked, or "" if
// allowed. Matching is on the base name so nested paths (certs/server.pem) are
// caught, while template files (.env.example) and unrelated names (staging.env)
// are not.
func denyWritePathReason(path string) string {
	if path == "" {
		return ""
	}
	base := filepath.Base(filepath.Clean(path))
	switch {
	case base == ".env":
		return "blocked by safety rule: writing .env"
	case strings.HasSuffix(base, ".pem"):
		return "blocked by safety rule: writing a .pem key"
	case strings.HasSuffix(base, ".key"):
		return "blocked by safety rule: writing a .key file"
	case strings.Contains(base, "credentials"):
		return "blocked by safety rule: writing a credentials file"
	}
	return ""
}

// shouldDenyToolCall checks if a tool call is blocked by the built-in safety
// rules. These apply in every permission mode.
func shouldDenyToolCall(toolName string, args json.RawMessage) (bool, string) {
	switch toolName {
	case "run_command":
		var input RunCommandInput
		if json.Unmarshal(args, &input) != nil {
			return false, ""
		}
		if reason := denyCommandReason(input.Command); reason != "" {
			return true, reason
		}
	case "write_file", "edit_file", "ast_edit":
		if reason := denyWritePathReason(extractMatchValue(toolName, args)); reason != "" {
			return true, reason
		}
	case "move_file":
		var input MoveFileInput
		if json.Unmarshal(args, &input) != nil {
			return false, ""
		}
		if reason := denyWritePathReason(input.Destination); reason != "" {
			return true, reason
		}
	}
	return false, ""
}

// describeToolCall generates a human-readable description of a tool call.
func describeToolCall(toolName string, args json.RawMessage) string {
	switch toolName {
	case "run_command":
		var input RunCommandInput
		if json.Unmarshal(args, &input) == nil {
			return "Run command: " + truncateStr(input.Command, 100)
		}
	case "write_file":
		var input WriteFileInput
		if json.Unmarshal(args, &input) == nil {
			return "Write file: " + input.Path + " (" + formatSize(len(input.Content)) + ")"
		}
	case "edit_file":
		var input EditFileInput
		if json.Unmarshal(args, &input) == nil {
			return "Edit file: " + input.Path
		}
	}
	return toolName
}

// formatSize formats byte count as human-readable.
func formatSize(bytes int) string {
	if bytes < 1024 {
		return fmt.Sprintf("%d bytes", bytes)
	}
	return fmt.Sprintf("%.1f KB", float64(bytes)/1024)
}
