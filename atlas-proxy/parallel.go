package main

import (
	"encoding/json"
	"fmt"
	"sync"
)

// ---------------------------------------------------------------------------
// Parallel execution — runs independent subtasks concurrently
// ---------------------------------------------------------------------------

// executeParallelTasks runs a list of planned tasks, respecting dependencies.
// Independent tasks (no shared depends_on) run in parallel goroutines.
// Dependent tasks wait for their dependencies to complete.
func executeParallelTasks(tasks []PlannedTask, ctx *AgentContext) []TaskStatus {
	// Build dependency graph
	taskMap := make(map[string]*PlannedTask)
	results := make(map[string]*TaskStatus)
	var mu sync.Mutex

	for i := range tasks {
		taskMap[tasks[i].ID] = &tasks[i]
		results[tasks[i].ID] = &TaskStatus{ID: tasks[i].ID, Status: "pending"}
	}

	// Topological sort — find execution order respecting dependencies
	order := topologicalSort(tasks)

	// Execute in waves: all tasks whose dependencies are met run concurrently
	completed := make(map[string]bool)
	var completedMu sync.Mutex

	for len(completed) < len(tasks) {
		// Find tasks ready to run (all deps completed)
		var ready []PlannedTask
		for _, id := range order {
			completedMu.Lock()
			if completed[id] {
				completedMu.Unlock()
				continue
			}
			completedMu.Unlock()

			task := taskMap[id]
			allDepsMet := true
			for _, dep := range task.DependsOn {
				completedMu.Lock()
				if !completed[dep] {
					allDepsMet = false
				}
				completedMu.Unlock()
				if !allDepsMet {
					break
				}
			}
			if allDepsMet {
				ready = append(ready, *task)
			}
		}

		if len(ready) == 0 {
			// No progress possible — circular dependency or all done
			break
		}

		// Execute ready tasks in parallel
		var wg sync.WaitGroup
		for _, task := range ready {
			wg.Add(1)
			go func(t PlannedTask) {
				defer wg.Done()

				ctx.Stream("text", map[string]string{
					"content": fmt.Sprintf("Starting task: %s — %s", t.ID, t.Description),
				})

				// Run a sub-agent loop for this task
				err := runSubTask(t, ctx)

				mu.Lock()
				if err != nil {
					results[t.ID].Status = "failed"
					results[t.ID].Error = err.Error()
				} else {
					results[t.ID].Status = "completed"
				}
				mu.Unlock()

				completedMu.Lock()
				completed[t.ID] = true
				completedMu.Unlock()
			}(task)
		}
		wg.Wait()
	}

	// Collect results in original order
	var statusList []TaskStatus
	for _, t := range tasks {
		mu.Lock()
		statusList = append(statusList, *results[t.ID])
		mu.Unlock()
	}

	return statusList
}

// runSubTask runs a single planned task by constructing a focused prompt
// and executing it through the agent loop.
func runSubTask(task PlannedTask, parentCtx *AgentContext) error {
	// Create a sub-context with the same configuration
	subCtx := NewAgentContext(parentCtx.WorkingDir, parentCtx.Tier)
	subCtx.InferenceURL = parentCtx.InferenceURL
	subCtx.SandboxURL = parentCtx.SandboxURL
	subCtx.LensURL = parentCtx.LensURL
	subCtx.V3URL = parentCtx.V3URL
	subCtx.Project = parentCtx.Project
	subCtx.PermissionMode = parentCtx.PermissionMode
	subCtx.YoloMode = parentCtx.YoloMode
	subCtx.StreamFn = parentCtx.StreamFn
	subCtx.PermissionFn = parentCtx.PermissionFn
	subCtx.MaxTurns = 15 // Sub-tasks get limited turns

	// Copy file read state from parent
	parentCtx.mu.Lock()
	for k, v := range parentCtx.FilesRead {
		subCtx.FilesRead[k] = v
	}
	for k, v := range parentCtx.FileReadTimes {
		subCtx.FileReadTimes[k] = v
	}
	parentCtx.mu.Unlock()

	// Build focused prompt
	prompt := fmt.Sprintf("Task: %s", task.Description)
	if len(task.Files) > 0 {
		prompt += fmt.Sprintf("\nFiles to create/modify: %s", joinStrings(task.Files))
	}

	return runAgentLoop(subCtx, prompt)
}

// topologicalSort returns task IDs in dependency-respecting order.
func topologicalSort(tasks []PlannedTask) []string {
	// Build adjacency list
	deps := make(map[string][]string)
	for _, t := range tasks {
		deps[t.ID] = t.DependsOn
	}

	var order []string
	visited := make(map[string]bool)
	temp := make(map[string]bool)

	var visit func(string)
	visit = func(id string) {
		if visited[id] || temp[id] {
			return
		}
		temp[id] = true
		for _, dep := range deps[id] {
			visit(dep)
		}
		temp[id] = false
		visited[id] = true
		order = append(order, id)
	}

	for _, t := range tasks {
		visit(t.ID)
	}

	return order
}

// joinStrings joins a string slice with ", ".
func joinStrings(ss []string) string {
	result := ""
	for i, s := range ss {
		if i > 0 {
			result += ", "
		}
		result += s
	}
	return result
}

// executePlanTasksTool is the full implementation of the plan_tasks tool executor.
// It replaces the placeholder in tools.go when parallel.go is loaded.
func executePlanTasksTool(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
	var input PlanTasksInput
	if err := json.Unmarshal(rawInput, &input); err != nil {
		return nil, fmt.Errorf("invalid input: %w", err)
	}

	results := executeParallelTasks(input.Tasks, ctx)

	out := PlanTasksOutput{Results: results}
	outBytes, _ := json.Marshal(out)

	// Check if any failed
	allPassed := true
	for _, r := range results {
		if r.Status != "completed" {
			allPassed = false
			break
		}
	}

	return &ToolResult{Success: allPassed, Data: outBytes}, nil
}
