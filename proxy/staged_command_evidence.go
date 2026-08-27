package main

// Placeholder for commit 2. Kept minimal so commit 1 builds and is reviewable
// on its own: staged evidence has no consumer yet, and both hooks answer no.

func stagedCommandSatisfied(ctx *AgentContext, want string) bool { return false }

func stagedCoverageSatisfied(ctx *AgentContext, path, hash string) bool { return false }
