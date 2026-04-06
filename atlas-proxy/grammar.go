package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// JSON Schema generation for constrained output
// ---------------------------------------------------------------------------

// buildToolCallSchema generates the JSON Schema that constrains model output
// to exactly one of: tool_call, text, or done.
//
// Passed to llama-server as the "json_schema" parameter, which forces
// every token through grammar-guided decoding. The model cannot produce
// invalid JSON — structural correctness is guaranteed.
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

// ---------------------------------------------------------------------------
// GBNF Grammar fallback
// ---------------------------------------------------------------------------

// buildGBNFGrammar generates a GBNF grammar string that constrains output
// to the same tool_call/text/done union. Used as fallback if llama-server's
// json_schema oneOf support is insufficient.
func buildGBNFGrammar() string {
	var sb strings.Builder

	// Root: one of the three response types
	sb.WriteString("root ::= tool-call | text-response | done-response\n\n")

	// Tool call
	toolNames := make([]string, 0, len(toolRegistry))
	for name := range toolRegistry {
		toolNames = append(toolNames, fmt.Sprintf(`"\"%s\""`, name))
	}

	sb.WriteString("tool-call ::= \"{\" ws ")
	sb.WriteString(`"\"type\"" ws ":" ws "\"tool_call\"" ws "," ws `)
	sb.WriteString(`"\"name\"" ws ":" ws tool-name ws "," ws `)
	sb.WriteString(`"\"args\"" ws ":" ws json-object ws `)
	sb.WriteString("\"}\"\n\n")

	// Tool name enum
	sb.WriteString("tool-name ::= ")
	sb.WriteString(strings.Join(toolNames, " | "))
	sb.WriteString("\n\n")

	// Text response
	sb.WriteString("text-response ::= \"{\" ws ")
	sb.WriteString(`"\"type\"" ws ":" ws "\"text\"" ws "," ws `)
	sb.WriteString(`"\"content\"" ws ":" ws json-string ws `)
	sb.WriteString("\"}\"\n\n")

	// Done response
	sb.WriteString("done-response ::= \"{\" ws ")
	sb.WriteString(`"\"type\"" ws ":" ws "\"done\"" ws "," ws `)
	sb.WriteString(`"\"summary\"" ws ":" ws json-string ws `)
	sb.WriteString("\"}\"\n\n")

	// JSON primitives
	sb.WriteString("json-object ::= \"{\" ws (json-pair (\",\" ws json-pair)*)? ws \"}\"\n")
	sb.WriteString("json-pair ::= json-string ws \":\" ws json-value\n")
	sb.WriteString("json-array ::= \"[\" ws (json-value (\",\" ws json-value)*)? ws \"]\"\n")
	sb.WriteString("json-value ::= json-string | json-number | json-object | json-array | \"true\" | \"false\" | \"null\"\n")
	sb.WriteString(`json-string ::= "\"" json-char* "\""` + "\n")
	sb.WriteString(`json-char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]` + "\n")
	sb.WriteString("json-number ::= \"-\"? [0-9]+ (\".\" [0-9]+)? ([eE] [\"+\\-\"]? [0-9]+)?\n")
	sb.WriteString("ws ::= [ \\t\\n]*\n")

	return sb.String()
}

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
