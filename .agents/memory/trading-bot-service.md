---
name: Trading bot standalone Python service
description: Why trading-bot/ is a plain top-level dir run by configureWorkflow instead of an artifact, and how to run/restart it.
---

`trading-bot/` (a cloned/adapted pure-Python aiohttp+WebSocket crypto signal bot) lives as a
sibling to `artifacts/`, `lib/`, `scripts/` in this monorepo — NOT registered as an artifact.

**Why:** none of the supported artifact kinds (react-vite, expo, openscad, slides, video-js) fit
a standalone Python aiohttp service with its own static frontend; forcing it into the artifact
system would fight the scaffolds rather than help.

**How to apply:**
- Run/restart it via the `Trading Bot` workflow (`configureWorkflow`/`WorkflowsRestart`), command
  `python3.12 trading-bot/server.py`, fixed port 8000 (not `$PORT` — this isn't proxy-routed).
- Python deps installed via `uv` into `.pythonlibs` at the workspace root (language: python-3.12).
- Requires the `GROQ_API_KEY` secret for the AI analyst; without it the server still runs (engine/
  dashboard work), just logs "GROQ_API_KEY not set — AI analysis disabled".
- Not reachable via the artifact preview path system since it's not an artifact; use
  `curl localhost:8000/...` from the shell to debug, or screenshot via the running port directly.
