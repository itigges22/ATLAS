package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// ---------------------------------------------------------------------------
// Project detection — identifies language, framework, and build commands
// ---------------------------------------------------------------------------

// detectProjectInfo scans the project root for config files and identifies
// the language, framework, and build commands.
func detectProjectInfo(projectDir string) *ProjectInfo {
	entries, err := os.ReadDir(projectDir)
	if err != nil {
		return nil
	}

	fileNames := make(map[string]bool)
	for _, e := range entries {
		if !e.IsDir() {
			fileNames[e.Name()] = true
		}
	}

	// Check each language/framework in priority order
	if info := detectNodeJS(projectDir, fileNames); info != nil {
		return info
	}
	if info := detectPython(projectDir, fileNames); info != nil {
		return info
	}
	if info := detectRust(fileNames); info != nil {
		return info
	}
	if info := detectGo(fileNames); info != nil {
		return info
	}
	if info := detectC(fileNames); info != nil {
		return info
	}
	if info := detectShell(projectDir, fileNames); info != nil {
		return info
	}

	return nil
}

// ---------------------------------------------------------------------------
// Language-specific detectors
// ---------------------------------------------------------------------------

func detectNodeJS(projectDir string, files map[string]bool) *ProjectInfo {
	if !files["package.json"] {
		return nil
	}

	info := &ProjectInfo{
		Language:     "nodejs",
		ConfigFiles:  []string{"package.json"},
		BuildCommand: "npm run build",
		DevCommand:   "npm run dev",
		TestCommand:  "npm test",
	}

	// Parse package.json for framework detection
	data, err := os.ReadFile(filepath.Join(projectDir, "package.json"))
	if err != nil {
		return info
	}

	var pkg struct {
		Dependencies    map[string]string `json:"dependencies"`
		DevDependencies map[string]string `json:"devDependencies"`
	}
	if err := json.Unmarshal(data, &pkg); err != nil {
		return info
	}

	allDeps := make(map[string]string)
	for k, v := range pkg.Dependencies {
		allDeps[k] = v
	}
	for k, v := range pkg.DevDependencies {
		allDeps[k] = v
	}

	// Detect framework
	if _, ok := allDeps["next"]; ok {
		info.Framework = "nextjs"
		info.BuildCommand = "npx next build"
	} else if _, ok := allDeps["react"]; ok {
		info.Framework = "react"
	} else if _, ok := allDeps["vue"]; ok {
		info.Framework = "vue"
	} else if _, ok := allDeps["express"]; ok {
		info.Framework = "express"
	}

	// Add detected config files
	for _, f := range []string{"tsconfig.json", "tailwind.config.ts", "tailwind.config.js",
		"postcss.config.js", "next.config.js", "next.config.ts", "vite.config.ts"} {
		if files[f] {
			info.ConfigFiles = append(info.ConfigFiles, f)
		}
	}

	if files["tsconfig.json"] {
		info.Language = "typescript"
	}

	return info
}

func detectPython(projectDir string, files map[string]bool) *ProjectInfo {
	hasPython := files["pyproject.toml"] || files["setup.py"] || files["requirements.txt"] || files["Pipfile"]
	if !hasPython {
		// Check for .py files
		entries, _ := filepath.Glob(filepath.Join(projectDir, "*.py"))
		if len(entries) == 0 {
			return nil
		}
	}

	info := &ProjectInfo{
		Language:    "python",
		ConfigFiles: []string{},
	}

	for _, f := range []string{"pyproject.toml", "setup.py", "requirements.txt", "Pipfile"} {
		if files[f] {
			info.ConfigFiles = append(info.ConfigFiles, f)
		}
	}

	// Detect framework from requirements or pyproject
	if data, err := os.ReadFile(filepath.Join(projectDir, "requirements.txt")); err == nil {
		content := strings.ToLower(string(data))
		if strings.Contains(content, "flask") {
			info.Framework = "flask"
			info.DevCommand = "flask run"
		} else if strings.Contains(content, "django") {
			info.Framework = "django"
			info.DevCommand = "python manage.py runserver"
		} else if strings.Contains(content, "fastapi") {
			info.Framework = "fastapi"
			info.DevCommand = "uvicorn main:app --reload"
		}
	}

	info.TestCommand = "python -m pytest"
	info.BuildCommand = "python -m py_compile *.py"

	return info
}

func detectRust(files map[string]bool) *ProjectInfo {
	if !files["Cargo.toml"] {
		return nil
	}
	return &ProjectInfo{
		Language:     "rust",
		ConfigFiles:  []string{"Cargo.toml"},
		BuildCommand: "cargo build",
		DevCommand:   "cargo run",
		TestCommand:  "cargo test",
	}
}

func detectGo(files map[string]bool) *ProjectInfo {
	if !files["go.mod"] {
		return nil
	}
	return &ProjectInfo{
		Language:     "go",
		ConfigFiles:  []string{"go.mod"},
		BuildCommand: "go build .",
		DevCommand:   "go run .",
		TestCommand:  "go test ./...",
	}
}

func detectC(files map[string]bool) *ProjectInfo {
	hasMakefile := files["Makefile"] || files["makefile"]
	hasCMake := files["CMakeLists.txt"]

	if !hasMakefile && !hasCMake {
		return nil
	}

	info := &ProjectInfo{
		Language:    "c",
		ConfigFiles: []string{},
	}

	if hasMakefile {
		info.ConfigFiles = append(info.ConfigFiles, "Makefile")
		info.BuildCommand = "make"
		info.TestCommand = "make test"
	}
	if hasCMake {
		info.ConfigFiles = append(info.ConfigFiles, "CMakeLists.txt")
		info.BuildCommand = "cmake --build ."
	}

	return info
}

func detectShell(projectDir string, files map[string]bool) *ProjectInfo {
	// Check for .sh files
	entries, _ := filepath.Glob(filepath.Join(projectDir, "*.sh"))
	if len(entries) == 0 {
		return nil
	}

	return &ProjectInfo{
		Language:     "shell",
		ConfigFiles:  []string{},
		BuildCommand: "bash -n *.sh",
		TestCommand:  "bash -n *.sh",
	}
}
