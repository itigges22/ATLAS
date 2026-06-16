package main

import (
	"encoding/json"
	"fmt"
	"os"
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
// passed to llama-server.
func buildToolCallSchema() map[string]interface{} {
	return buildToolCallSchemaForTools(nil)
}

// buildToolCallSchemaForTools is the same as buildToolCallSchema, but with
// the named tools removed from the tool-name enum. Used by callLLMConstrained
// to ban edit_file/write_file for a single decision step after a write_file
// rejection on .py/.html (BiasBusters #3, May 2026 research synthesis).
func buildToolCallSchemaForTools(excluded []string) map[string]interface{} {
	excludeSet := make(map[string]struct{}, len(excluded))
	for _, name := range excluded {
		excludeSet[name] = struct{}{}
	}
	toolNames := make([]interface{}, 0, len(toolRegistry))
	for name := range toolRegistry {
		if _, skip := excludeSet[name]; skip {
			continue
		}
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

// buildToolCallSchemaJSONForTools returns the JSON-encoded schema with the
// listed tools excluded from the tool-name enum.
func buildToolCallSchemaJSONForTools(excluded []string) string {
	schema := buildToolCallSchemaForTools(excluded)
	b, _ := json.Marshal(schema)
	return string(b)
}

// buildResponseFormat picks the response_format payload to send to
// llama-server based on ATLAS_GRAMMAR_MODE (#33).
//
//	"strict" (default): {"type":"json_object","schema":<full schema>}.
//	  llama-server converts the schema to internal GBNF at the C side
//	  so the token sampler can ONLY emit our tool_call/text/done union.
//	  Previously the model could emit any valid JSON and we'd reject
//	  + retry post-hoc, burning tokens; the schema-constrained path
//	  eliminates that whole class of waste.
//
//	"loose": {"type":"json_object"} — old behavior, "valid JSON only,
//	  shape not enforced." Kept as an escape hatch in case a model
//	  handles the schema-to-GBNF conversion poorly (rare, but a
//	  one-env-var rollback beats a code revert).
//
// Returns an interface{} because the strict case nests a map (the
// schema), which doesn't fit map[string]string.
func buildResponseFormat() interface{} {
	mode := os.Getenv("ATLAS_GRAMMAR_MODE")
	if mode == "" {
		mode = "strict"
	}
	if mode == "loose" {
		return map[string]string{"type": "json_object"}
	}
	return map[string]interface{}{
		"type":   "json_object",
		"schema": buildToolCallSchema(),
	}
}

// ---------------------------------------------------------------------------
// GBNF Grammar fallback
// ---------------------------------------------------------------------------

// buildGBNFGrammar generates a GBNF grammar string that constrains output
// to the same tool_call/text/done union. Currently unused; kept as
// reference in case json_object mode needs to be replaced with GBNF.
func buildGBNFGrammar() string {
	return buildGBNFGrammarForTools(nil)
}

// buildGBNFGrammarForTools is buildGBNFGrammar with the listed tools
// removed from the tool-name production. May 2026 BiasBusters #2: when
// the next step must NOT use edit_file (e.g. write_file just got rejected
// on a .py/.html file >5 lines), llama-server enforces the restriction
// at the token-decode level via the `grammar` parameter — descriptions
// and system prompt rules can be ignored by the model, but the grammar
// physically cannot emit the banned tool name.
func buildGBNFGrammarForTools(excluded []string) string {
	excludeSet := make(map[string]struct{}, len(excluded))
	for _, name := range excluded {
		excludeSet[name] = struct{}{}
	}

	var sb strings.Builder

	// Root: one of the three response types
	sb.WriteString("root ::= tool-call | text-response | done-response\n\n")

	// Tool call
	toolNames := make([]string, 0, len(toolRegistry))
	for name := range toolRegistry {
		if _, skip := excludeSet[name]; skip {
			continue
		}
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
	return buildToolDescriptionsExcluding(nil)
}

// buildToolDescriptionsExcluding generates the tool documentation section
// with the listed tools removed entirely. Used for the per-step nudge
// note when edit_file/write_file must be banned for a single decision.
// The system prompt is built once per session and we don't rebuild it,
// but this is reused inside buildStepAdvisoryNote() to remind the model
// of the available palette without the banned tool present.
func buildToolDescriptionsExcluding(excluded []string) string {
	excludeSet := make(map[string]struct{}, len(excluded))
	for _, name := range excluded {
		excludeSet[name] = struct{}{}
	}
	var sb strings.Builder
	sb.WriteString("## Available Tools\n\n")
	sb.WriteString("You must respond with a JSON object in one of these formats:\n\n")
	sb.WriteString("**Tool call:** `{\"type\":\"tool_call\",\"name\":\"<tool>\",\"args\":{...}}`\n")
	sb.WriteString("**Text message:** `{\"type\":\"text\",\"content\":\"<message>\"}`\n")
	sb.WriteString("**Task complete:** `{\"type\":\"done\",\"summary\":\"<what you did>\"}`\n\n")

	for _, tool := range allTools() {
		if _, skip := excludeSet[tool.Name]; skip {
			continue
		}
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
	case "outline_file":
		return `{"path": "src/main.py"}`
	case "write_file":
		return `{"path": "src/main.py", "content": "#!/usr/bin/env python3\n..."}`
	case "edit_file":
		// Real fix-style snippet — adding a None check, the most common
		// kind of small targeted edit. Models cargo-cult the example
		// shape, so a "rename foo to bar" placeholder steered them
		// toward purely cosmetic edits instead of real bug-fix shapes.
		return `{"path": "src/main.py", "old_str": "if x == 0:\n        return None", "new_str": "if x is None or x == 0:\n        return None", "replace_all": false}`
	case "ast_edit":
		// Whole-function rewrite — the case where edit_file would force
		// the model to copy the entire existing function as old_str and
		// blow through max_tokens. Selector grammar is intentionally
		// narrow in v1 (function:NAME, class:NAME, <tag>) to avoid the
		// raw-tree-sitter hallucination problem (GH #39 measurement).
		return `{"path": "src/main.py", "selector": "function:dashboard", "content": "@app.route('/dashboard')\ndef dashboard():\n    return render_template('dashboard.html')"}`
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
