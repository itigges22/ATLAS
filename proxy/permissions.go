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

// PermissionConfig holds the loaded permission rules.
type PermissionConfig struct {
	AllowRules []PermissionRule
	DenyRules  []PermissionRule
}

// DefaultDenyPatterns are always blocked regardless of mode.
var DefaultDenyPatterns = []PermissionRule{
	{Tool: "run_command", Pattern: "rm -rf /", Action: "deny"},
	{Tool: "run_command", Pattern: "rm -rf /*", Action: "deny"},
	{Tool: "run_command", Pattern: "mkfs*", Action: "deny"},
	{Tool: "run_command", Pattern: "dd if=*of=/dev/*", Action: "deny"},
	{Tool: "write_file", Pattern: ".env", Action: "deny"},
	{Tool: "write_file", Pattern: "*.pem", Action: "deny"},
	{Tool: "write_file", Pattern: "*.key", Action: "deny"},
	{Tool: "write_file", Pattern: "*credentials*", Action: "deny"},
}

// checkPermissionRules evaluates rules against a tool call.
// Returns "allow", "deny", or "" (no matching rule).
func checkPermissionRules(rules []PermissionRule, toolName string, args json.RawMessage) string {
	for _, rule := range rules {
		if rule.Tool != toolName {
			continue
		}

		// Extract the relevant value from args for pattern matching
		matchValue := extractMatchValue(toolName, args)
		if matchValue == "" {
			continue
		}

		// Match pattern against value
		if matchPattern(rule.Pattern, matchValue) {
			return rule.Action
		}
	}
	return ""
}

// extractMatchValue extracts the value to match against permission patterns.
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
	}
	return ""
}

// matchPattern matches a glob-like pattern against a value.
// Supports * as wildcard.
func matchPattern(pattern, value string) bool {
	// Direct match
	if pattern == value {
		return true
	}

	// Glob-style matching
	matched, err := filepath.Match(pattern, value)
	if err == nil && matched {
		return true
	}

	// Check if pattern is a prefix with wildcard
	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		if strings.HasPrefix(value, prefix) {
			return true
		}
	}

	// Check if value contains the pattern (for command matching)
	if strings.Contains(value, pattern) {
		return true
	}

	return false
}

// shouldDenyToolCall checks if a tool call should be blocked by deny rules.
func shouldDenyToolCall(toolName string, args json.RawMessage) (bool, string) {
	// Check default deny patterns
	for _, rule := range DefaultDenyPatterns {
		if rule.Tool != toolName {
			continue
		}
		matchValue := extractMatchValue(toolName, args)
		if matchValue != "" && matchPattern(rule.Pattern, matchValue) {
			return true, "blocked by safety rule: " + rule.Pattern
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
