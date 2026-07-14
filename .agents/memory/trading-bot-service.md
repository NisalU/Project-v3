---
name: Trading bot standalone Python service
description: Why trading-bot/ is a plain top-level dir run by a workflow instead of a pnpm artifact, and how to run/restart it.
---

`trading-bot/` (a pure-Python aiohttp+WebSocket crypto signal bot, imported from an external repo)
lives as a sibling to `artifacts/`, `lib/`, `scripts/` in this monorepo — NOT registered as an
artifact.

**Why:** none of the supported artifact kinds (react-vite, expo, openscad, slides, video-js) fit a
standalone Python aiohttp service with its own static frontend; forcing it into the artifact
system would fight the scaffolds rather than help.

**How to apply:**
- Run/restart it via the `Trading Bot` workflow (`configureWorkflow`/`WorkflowsRestart`), command
  `python3.12 trading-bot/server.py`, fixed port 8000 (not `$PORT` — this isn't proxy-routed).
- Python deps installed via `uv` into `.pythonlibs` at the workspace root (language: python-3.12).
- Requires the `GROQ_API_KEY` secret for the AI analyst; without it the server still runs (engine/
  dashboard work), just logs "GROQ_API_KEY not set — AI analysis disabled".
- Not reachable via the artifact preview path system since it's not an artifact; use
  `curl localhost:8000/...` from the shell to debug, or view it via its own port directly. The
  generic `Screenshot` tool also can't reach it (no `artifactDirName`, and `externalUrl` rejects
  localhost) — verify via curl/logs instead.

**Dashboard WS client quirk:** each new `Client` on the server starts pinned to
`config.DEFAULT_SYMBOL`/`DEFAULT_INTERVAL` until it receives a `subscribe` message; the frontend
must send one right after processing the initial `config` message (not just rely on `onopen`,
which fires before `config` arrives and the `<select>` is still empty) — otherwise a
restored/non-default symbol shows in the dropdown while the chart/data silently stays on the
server's default market.
