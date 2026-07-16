# ATLAS VS Code Extension

A VS Code client for the [ATLAS](https://github.com/itigges22/ATLAS) agent proxy — a thin UI layer wrapping `atlas-proxy`'s agent loop (chat, tool calls, permission gating, diffs) with no agent logic in the extension itself.

**Status: Work in progress.** Tracking [issue #35](https://github.com/itigges22/ATLAS/issues/35). Currently scaffold-only — chat UI, permission flow, and diff rendering land in upcoming commits.

## How it works

The extension is a thin SSE client over the proxy HTTP API (see `docs/API.md`):

* `POST /v1/agent` — streams a turn (text tokens, tool calls/results, permission requests) as server-sent events
* `POST /v1/permission` — answers permission requests raised mid-turn
* `POST /cancel` — cancels the in-flight turn
* `GET /ready` — status bar connectivity polling

The TUI (`tui/`) is the reference client; the extension mirrors its session conventions (client-minted `session_id` per turn, `session_allowed_tools` re-sent each turn, cancel = abort + best-effort `POST /cancel`).

## Settings

* `atlas.proxyUrl` — base URL of the ATLAS proxy server (default `http://localhost:8090`)
* `atlas.serviceToken` — dev-override bearer token (prefer the `ATLAS: Set Service Token` command, which stores it in SecretStorage instead of plaintext settings)
* `atlas.permissionMode` — `default` / `accept-edits` / `yolo`
* `atlas.statusBar.enabled`, `atlas.statusBar.pollIntervalSec` — status bar connectivity polling

## Commands

* `ATLAS: Open Chat`
* `ATLAS: Cancel Current Turn`
* `ATLAS: Set Service Token`
* `ATLAS: New Conversation`

## Known limitations

* The proxy applies tool calls to its own mounted workspace (`ATLAS_WORKSPACE_DIR`). If the VS Code workspace folder is not the same directory, edits land elsewhere — the extension detects the mismatch heuristically and warns, but cannot verify the proxy's mount (no endpoint exposes it yet).

## Development

```bash
npm install
npm run compile   # type-check + lint + bundle
npm test          # vitest unit tests
```

Press `F5` in VS Code to launch an Extension Development Host for testing against a live proxy (`atlas up`).

## Layout

```
src/
├── extension.ts      # activate(), command registration
├── client/           # proxy HTTP/SSE client (atlasClient, sse parser, API types)
├── session/          # turn lifecycle (session_id, history, permission flow)
├── ui/               # chat webview, status bar, diff provider
├── workspace/        # workspace-mismatch heuristic
└── util/             # error-envelope mapping
media/                # webview assets
test/                 # vitest unit tests + mock proxy fixture
```
