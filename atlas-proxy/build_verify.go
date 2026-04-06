package main

import (
	"path/filepath"
	"strings"
)

// ---------------------------------------------------------------------------
// Build verification — per-file-type verification commands
// ---------------------------------------------------------------------------

// BuildVerification holds the verification strategy for a file.
type BuildVerification struct {
	// Commands to run for verification (in order). All must pass.
	Commands []string
	// Description of what's being verified.
	Description string
}

// getVerificationForFile returns the appropriate build verification
// commands for a given file path and project context.
func getVerificationForFile(filePath string, project *ProjectInfo, workingDir string) *BuildVerification {
	ext := strings.ToLower(filepath.Ext(filePath))
	base := filepath.Base(filePath)

	// Framework-specific overrides first
	if project != nil {
		if v := frameworkVerification(filePath, project); v != nil {
			return v
		}
	}

	switch ext {
	case ".ts", ".tsx":
		return &BuildVerification{
			Commands:    []string{"npx tsc --noEmit"},
			Description: "TypeScript type check",
		}

	case ".js", ".jsx":
		return &BuildVerification{
			Commands:    []string{"node --check " + filePath},
			Description: "JavaScript syntax check",
		}

	case ".py":
		module := strings.TrimSuffix(base, ext)
		return &BuildVerification{
			Commands: []string{
				"python -m py_compile " + filePath,
				"python -c \"import " + module + "\"",
			},
			Description: "Python compile + import check",
		}

	case ".c", ".h":
		if project != nil && project.BuildCommand == "make" {
			return &BuildVerification{
				Commands:    []string{"make"},
				Description: "Make build",
			}
		}
		return &BuildVerification{
			Commands:    []string{"gcc -fsyntax-only " + filePath},
			Description: "C syntax check",
		}

	case ".cpp", ".cc", ".cxx", ".hpp":
		if project != nil && project.BuildCommand == "make" {
			return &BuildVerification{
				Commands:    []string{"make"},
				Description: "Make build",
			}
		}
		return &BuildVerification{
			Commands:    []string{"g++ -fsyntax-only " + filePath},
			Description: "C++ syntax check",
		}

	case ".go":
		return &BuildVerification{
			Commands:    []string{"go build ."},
			Description: "Go build",
		}

	case ".rs":
		return &BuildVerification{
			Commands:    []string{"cargo check"},
			Description: "Rust cargo check",
		}

	case ".sh", ".bash":
		return &BuildVerification{
			Commands:    []string{"bash -n " + filePath},
			Description: "Shell syntax check",
		}

	case ".json":
		return &BuildVerification{
			Commands:    []string{"python -c \"import json; json.load(open('" + filePath + "'))\""},
			Description: "JSON validation",
		}

	case ".toml":
		return &BuildVerification{
			Commands:    []string{"python -c \"import tomllib; tomllib.load(open('" + filePath + "', 'rb'))\""},
			Description: "TOML validation",
		}

	default:
		return nil
	}
}

// frameworkVerification returns framework-specific verification when available.
func frameworkVerification(filePath string, project *ProjectInfo) *BuildVerification {
	switch project.Framework {
	case "nextjs":
		ext := filepath.Ext(filePath)
		if ext == ".ts" || ext == ".tsx" || ext == ".js" || ext == ".jsx" {
			return &BuildVerification{
				Commands:    []string{"npx tsc --noEmit"},
				Description: "Next.js TypeScript check",
			}
		}

	case "flask":
		if filepath.Ext(filePath) == ".py" {
			// Try importing the Flask app
			return &BuildVerification{
				Commands: []string{
					"python -m py_compile " + filePath,
				},
				Description: "Flask Python compile check",
			}
		}

	case "express":
		if filepath.Ext(filePath) == ".js" || filepath.Ext(filePath) == ".ts" {
			return &BuildVerification{
				Commands:    []string{"node --check " + filePath},
				Description: "Express syntax check",
			}
		}
	}

	return nil
}

// getFullBuildCommand returns the project's build command for full project verification.
func getFullBuildCommand(project *ProjectInfo) string {
	if project == nil || project.BuildCommand == "" {
		return ""
	}
	return project.BuildCommand
}
