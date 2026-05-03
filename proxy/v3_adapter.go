package main

import (
	"fmt"
	"path/filepath"
	"strings"
)

// ---------------------------------------------------------------------------
// V3 Adapter — translates arbitrary file requests into V3 pipeline format
// ---------------------------------------------------------------------------

// buildV3Request constructs a V3GenerateRequest from an agent context
// and a write_file call. This adapts arbitrary coding tasks into the
// format the V3 Python service expects.
func buildV3Request(filePath, baselineContent string, ctx *AgentContext) V3GenerateRequest {
	req := V3GenerateRequest{
		FilePath:     filePath,
		BaselineCode: baselineContent,
		Tier:         int(ctx.Tier),
		WorkingDir:   ctx.WorkingDir,
	}

	// Add project context from files read during this session
	if len(ctx.FilesRead) > 0 {
		req.ProjectContext = make(map[string]string)
		for p, content := range ctx.FilesRead {
			relPath, err := filepath.Rel(ctx.WorkingDir, p)
			if err != nil {
				relPath = p
			}
			// Truncate large files to save tokens in V3 pipeline
			if len(content) > 4000 {
				content = content[:4000] + "\n... (truncated)"
			}
			req.ProjectContext[relPath] = content
		}
	}

	// Add framework and build command from project detection
	if ctx.Project != nil {
		req.Framework = ctx.Project.Framework
		req.BuildCommand = ctx.Project.BuildCommand
	}

	// Extract constraints from file type and project context
	req.Constraints = extractConstraints(filePath, baselineContent, ctx)

	return req
}

// extractConstraints derives constraints from the file path, content, and project.
// These guide PlanSearch and DivSampling toward framework-appropriate solutions.
func extractConstraints(filePath, content string, ctx *AgentContext) []string {
	var constraints []string
	ext := filepath.Ext(filePath)
	base := filepath.Base(filePath)

	// File-type constraints
	switch ext {
	case ".tsx", ".jsx":
		constraints = append(constraints, "Must be a valid React component")
		constraints = append(constraints, "Must use proper JSX syntax")
		if strings.Contains(content, "use client") || strings.Contains(content, "'use client'") {
			constraints = append(constraints, "Client component — must have 'use client' directive")
		}
	case ".ts":
		constraints = append(constraints, "Must be valid TypeScript")
	case ".py":
		constraints = append(constraints, "Must be valid Python 3")
	case ".go":
		constraints = append(constraints, "Must be valid Go with proper package declaration")
	case ".rs":
		constraints = append(constraints, "Must be valid Rust")
	case ".c", ".h":
		constraints = append(constraints, "Must be valid C (C11 or later)")
	case ".sh":
		constraints = append(constraints, "Must be a valid bash script")
		constraints = append(constraints, "Must start with #!/bin/bash or #!/usr/bin/env bash")
	}

	// Framework constraints
	if ctx.Project != nil {
		switch ctx.Project.Framework {
		case "nextjs":
			constraints = append(constraints, "Must follow Next.js 14 App Router conventions")
			if strings.Contains(filePath, "app/api/") {
				constraints = append(constraints, "Must export HTTP method handlers (GET, POST, etc.)")
				constraints = append(constraints, "Must use NextResponse for responses")
			}
			if strings.Contains(filePath, "app/") && (ext == ".tsx" || ext == ".jsx") {
				constraints = append(constraints, "Must be a valid Next.js page or component")
			}
		case "flask":
			constraints = append(constraints, "Must follow Flask conventions")
			if base == "app.py" || base == "main.py" {
				constraints = append(constraints, "Must create a Flask app instance")
			}
		case "express":
			constraints = append(constraints, "Must follow Express.js conventions")
		}
	}

	// Import constraints from project context
	if ctx.FilesRead != nil {
		// If we've read other files, ensure new file imports are consistent
		for readPath := range ctx.FilesRead {
			relRead, _ := filepath.Rel(ctx.WorkingDir, readPath)
			relFile, _ := filepath.Rel(ctx.WorkingDir, filePath)
			if relRead != "" && relFile != "" {
				dir := filepath.Dir(relFile)
				readDir := filepath.Dir(relRead)
				if dir == readDir {
					constraints = append(constraints,
						fmt.Sprintf("Must be consistent with sibling file %s", filepath.Base(readPath)))
				}
			}
		}
	}

	return constraints
}

// buildPromptFromRequest constructs the problem description that the V3 pipeline
// will use for PlanSearch constraint extraction and candidate generation.
// This replaces the benchmark's task.prompt with a rich description of what
// the file should contain, including project context.
func buildPromptFromRequest(req V3GenerateRequest) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Create the file `%s`", req.FilePath))

	if req.Framework != "" {
		sb.WriteString(fmt.Sprintf(" for a %s project", req.Framework))
	}
	sb.WriteString(".\n\n")

	// Include project context
	if len(req.ProjectContext) > 0 {
		sb.WriteString("## Existing project files:\n\n")
		for path, content := range req.ProjectContext {
			// Show short files in full, summarize long ones
			if len(content) < 500 {
				sb.WriteString(fmt.Sprintf("### %s\n```\n%s\n```\n\n", path, content))
			} else {
				// Show first 300 chars as summary
				sb.WriteString(fmt.Sprintf("### %s (truncated)\n```\n%s\n...\n```\n\n",
					path, content[:300]))
			}
		}
	}

	// Include constraints
	if len(req.Constraints) > 0 {
		sb.WriteString("## Requirements:\n")
		for _, c := range req.Constraints {
			sb.WriteString(fmt.Sprintf("- %s\n", c))
		}
		sb.WriteString("\n")
	}

	// Include build command
	if req.BuildCommand != "" {
		sb.WriteString(fmt.Sprintf("## Build verification:\nThe file must pass: `%s`\n\n", req.BuildCommand))
	}

	// Include baseline content as reference
	if req.BaselineCode != "" {
		sb.WriteString("## Reference implementation:\n")
		sb.WriteString("The following is a baseline implementation. Improve upon it if possible, ")
		sb.WriteString("but ensure all functionality is preserved.\n\n")
		sb.WriteString(fmt.Sprintf("```\n%s\n```\n", req.BaselineCode))
	}

	return sb.String()
}
