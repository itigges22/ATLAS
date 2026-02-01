# Task: Improve GitHub Repository Documentation

## CRITICAL INSTRUCTION
You MUST explore the actual codebase before writing any documentation. Do NOT assume or guess the architecture. Read the source files to understand what each service does.

## Phase 1: Understand the Codebase

Before writing anything, run these commands and READ the output:
```bash
# 1. List all services
ls -la /home/nobase/k8s/

# 2. Read each service's main file to understand what it does
cat /home/nobase/k8s/llama-server/Dockerfile
cat /home/nobase/k8s/llama-server/entrypoint.sh

cat /home/nobase/k8s/rag-api/main.py | head -100
cat /home/nobase/k8s/rag-api/rag.py | head -100

cat /home/nobase/k8s/api-portal/src/main.py | head -100

cat /home/nobase/k8s/llm-proxy/main.py | head -100

cat /home/nobase/k8s/task-worker/main.py | head -100
cat /home/nobase/k8s/task-worker/ralph_loop.py | head -100

cat /home/nobase/k8s/sandbox/main.py | head -100

cat /home/nobase/k8s/dashboard/server.js | head -100 2>/dev/null || cat /home/nobase/k8s/dashboard/main.py | head -100

cat /home/nobase/k8s/embedding-service/server.py | head -100

# 3. Read the templates to understand service connections
ls /home/nobase/k8s/templates/
cat /home/nobase/k8s/templates/rag-api-deployment.yaml.tmpl
cat /home/nobase/k8s/templates/task-worker-deployment.yaml.tmpl

# 4. Read the config to understand all configurable components
cat /home/nobase/k8s/atlas.conf.example

# 5. Check what tests exist to understand functionality
ls /home/nobase/k8s/tests/infrastructure/
ls /home/nobase/k8s/tests/integration/

# 6. Read the current README and docs to see what needs fixing
cat /home/nobase/k8s/README.md
cat /home/nobase/k8s/docs/ARCHITECTURE.md
cat /home/nobase/k8s/CONTRIBUTING.md
```

## Phase 2: Create CODE_OF_CONDUCT.md

Create a separate `/home/nobase/k8s/CODE_OF_CONDUCT.md` file using the Contributor Covenant v2.1 (standard for open source). 

Remove any code of conduct content from CONTRIBUTING.md if it's embedded there.

## Phase 3: Rewrite README.md

Study this example first:
```bash
# Fetch Tabby's README for reference
curl -s https://raw.githubusercontent.com/TabbyML/tabby/main/README.md | head -200
```

The new README.md must:

1. **Be SHORT and visual** - Not walls of text
2. **Have badges at the top** - GitHub stars, license, tests passing, Discord (if applicable)
3. **Have a hero section** - One-liner + key selling points as icons/bullets
4. **Include a GIF or screenshot** - Of the system in action (placeholder for now)
5. **Have "Getting Started" in 5 lines max**
6. **Link to docs for details** - Don't put technical details in README
7. **Have a professional architecture diagram** - See Phase 4
8. **Include sections**: Features, Quick Start, Documentation, Community, License
9. **NOT include**: Detailed configuration tables, troubleshooting, API endpoints

The tone should be:
- Marketing-friendly, not developer-docs
- "What can this do for me?" not "Here's how it works internally"
- Clean, scannable, visual

## Phase 4: Create Professional Architecture Diagram

Create a Mermaid diagram that can be rendered by GitHub. The diagram must:

1. Be based on ACTUAL code you read in Phase 1
2. Show the real data flow between services
3. Be visually clean and not cluttered
4. Include these components (verify they exist first):
   - External client
   - llm-proxy (auth, rate limiting)
   - rag-api (RAG orchestration)
   - llama-server (LLM inference)
   - task-worker (Ralph Loop)
   - sandbox (code execution)
   - embedding-service
   - qdrant (vector DB)
   - redis (queues, cache)
   - api-portal (user management)
   - dashboard (monitoring)
   - training pipeline (if exists)

Create the diagram in two places:
1. Inline in README.md using ```mermaid code block
2. As a separate file: /home/nobase/k8s/docs/architecture-diagram.md

## Phase 5: Rewrite docs/ARCHITECTURE.md

This document should contain the DETAILED technical explanation that was removed from README. Include:

1. **System Overview** - High-level description
2. **Component Deep Dive** - Each service explained with:
   - Purpose
   - Technology used
   - Ports
   - How it connects to other services
3. **Data Flows** - Explain the path of:
   - A chat completion request
   - A coding task through Ralph Loop
   - Project sync for RAG
   - Training data collection
4. **Ralph Loop Algorithm** - Detailed explanation with pseudocode
5. **Continuous Learning Pipeline** - How training works
6. **RAG Pipeline** - How code context is retrieved

Base ALL of this on the actual code you read in Phase 1.

## Phase 6: Update CONTRIBUTING.md

Remove any Code of Conduct content (now in separate file). Keep only:
- How to report issues
- How to submit PRs
- Code style guidelines
- Testing requirements
- Development setup

## Verification

After completing all phases:
```bash
# Verify files exist
ls -la /home/nobase/k8s/CODE_OF_CONDUCT.md
ls -la /home/nobase/k8s/README.md
ls -la /home/nobase/k8s/docs/ARCHITECTURE.md
ls -la /home/nobase/k8s/CONTRIBUTING.md

# Verify no code of conduct in contributing
grep -i "code of conduct" /home/nobase/k8s/CONTRIBUTING.md

# Verify mermaid diagram in README
grep -A 50 '```mermaid' /home/nobase/k8s/README.md

# Check README length (should be concise)
wc -l /home/nobase/k8s/README.md
# Target: Under 200 lines

# Check ARCHITECTURE.md has detail
wc -l /home/nobase/k8s/docs/ARCHITECTURE.md
# Target: 300+ lines with real technical content
```

## Success Criteria

- [ ] CODE_OF_CONDUCT.md exists as separate file (Contributor Covenant v2.1)
- [ ] README.md is under 200 lines
- [ ] README.md has badges, hero section, mermaid diagram
- [ ] README.md style matches Tabby's (clean, visual, marketing-friendly)
- [ ] README.md does NOT have detailed config tables or troubleshooting
- [ ] Mermaid architecture diagram accurately reflects actual codebase
- [ ] docs/ARCHITECTURE.md has 300+ lines of real technical detail
- [ ] docs/ARCHITECTURE.md explains Ralph Loop, RAG, Training pipelines
- [ ] CONTRIBUTING.md has no embedded Code of Conduct
- [ ] All architecture info is based on actual code, not assumptions

## Begin

Start with Phase 1 - read the codebase thoroughly before writing anything.
