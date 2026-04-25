package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// JSON Schema generation for constrained output
// ---------------------------------------------------------------------------

// buildToolCallSchema generates the JSON Schema that describes the valid output
// format: exactly one of tool_call, text, or done.
//
// The actual constraint is enforced by response_format: json_object in the
// LLM request. This schema is available for reference but not directly
// passed to vLLM gen instance.
func buildToolCallSchema() map[string]interface{} {
	toolNames := make([]interface{}, 0, len(toolRegistry))
	for name := range toolRegistry {
		toolNames = append(toolNames, name)
	}

	return map[string]interface{}{
		"oneOf": []interface{}{
			// Tool call variant
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"type": map[string]interface{}{
						"type": "string",
						"enum": []string{"tool_call"},
					},
					"name": map[string]interface{}{
						"type": "string",
						"enum": toolNames,
					},
					"args": map[string]interface{}{
						"type": "object",
					},
				},
				"required":             []string{"type", "name", "args"},
				"additionalProperties": false,
			},
			// Text variant
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"type": map[string]interface{}{
						"type": "string",
						"enum": []string{"text"},
					},
					"content": map[string]interface{}{
						"type": "string",
					},
				},
				"required":             []string{"type", "content"},
				"additionalProperties": false,
			},
			// Done variant
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"type": map[string]interface{}{
						"type": "string",
						"enum": []string{"done"},
					},
					"summary": map[string]interface{}{
						"type": "string",
					},
				},
				"required":             []string{"type", "summary"},
				"additionalProperties": false,
			},
		},
	}
}

// buildToolCallSchemaJSON returns the JSON-encoded schema string.
func buildToolCallSchemaJSON() string {
	schema := buildToolCallSchema()
	b, _ := json.Marshal(schema)
	return string(b)
}

// vLLM uses `response_format: {"type": "json_object"}` (with a JSON Schema
// in `guided_json` for stricter shapes) — not the llama.cpp GBNF format.
// The previous codebase carried a `buildGBNFGrammar()` reference here; it
// was deleted in the vLLM cutover because GBNF is incompatible with vLLM
// and the function provided no starting point for the lark-format
// alternative if `guided_json` ever needs replacing.

// ---------------------------------------------------------------------------
// System prompt: tool descriptions for the model
// ---------------------------------------------------------------------------

// buildToolDescriptions generates the tool documentation section of the system prompt.
func buildToolDescriptions() string {
	var sb strings.Builder
	sb.WriteString("## Available Tools\n\n")
	sb.WriteString("You must respond with a JSON object in one of these formats:\n\n")
	sb.WriteString("**Tool call:** `{\"type\":\"tool_call\",\"name\":\"<tool>\",\"args\":{...}}`\n")
	sb.WriteString("**Text message:** `{\"type\":\"text\",\"content\":\"<message>\"}`\n")
	sb.WriteString("**Task complete:** `{\"type\":\"done\",\"summary\":\"<what you did>\"}`\n\n")

	for _, tool := range allTools() {
		sb.WriteString(fmt.Sprintf("### %s\n", tool.Name))
		sb.WriteString(fmt.Sprintf("%s\n\n", tool.Description))
		sb.WriteString("**Input:**\n```json\n")

		// Generate example from input schema struct
		schemaJSON := generateInputExample(tool.Name)
		sb.WriteString(schemaJSON)
		sb.WriteString("\n```\n\n")
	}

	return sb.String()
}

// generateInputExample creates an example JSON for a tool's input.
func generateInputExample(toolName string) string {
	switch toolName {
	case "read_file":
		return `{"path": "src/main.py", "offset": 0, "limit": 100}`
	case "write_file":
		return `{"path": "src/main.py", "content": "#!/usr/bin/env python3\n..."}`
	case "edit_file":
		return `{"path": "src/main.py", "old_str": "def foo():", "new_str": "def bar():", "replace_all": false}`
	case "delete_file":
		return `{"path": "old_file.py"}`
	case "run_command":
		return `{"command": "python -m py_compile src/main.py", "timeout": 30}`
	case "search_files":
		return `{"pattern": "def main", "path": "src/", "glob": "*.py"}`
	case "list_directory":
		return `{"path": "."}`
	case "plan_tasks":
		return `{"tasks": [{"id": "config", "description": "Create config files", "files": ["package.json"], "depends_on": []}]}`
	default:
		return `{}`
	}
}
