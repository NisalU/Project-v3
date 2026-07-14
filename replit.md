# Project-v3

A crypto trading signal bot with a live charting dashboard, plus a pnpm workspace scaffold for future web apps.

## Run & Operate

- Trading bot: runs via the `Trading Bot` workflow (`python3.12 trading-bot/server.py`), served on port 8000
- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- Optional env: `GROQ_API_KEY` — enables the trading bot's AI trade analyst; without it the bot still runs (charts/signals work), AI analysis is just disabled

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)
- Trading bot: standalone Python 3.12 + aiohttp/WebSocket service (no numpy/pandas)

## Where things live

- `trading-bot/` — standalone Python crypto signal bot (not a pnpm artifact); `server.py` is the entrypoint, `engine.py` + `strategies/` compute the composite score, `ai_analyst.py` + `trade_tracker.py` manage the AI trade lifecycle, `static/` is the dashboard frontend
- _Populate the rest as you build the pnpm-workspace side — repo map plus pointers to the source-of-truth file for DB schema, API contracts, theme files, etc._

## Architecture decisions

- The trading bot lives as a plain top-level `trading-bot/` directory (sibling to `artifacts/`), not registered as a pnpm artifact — none of the artifact scaffolds (react-vite, expo, etc.) fit a standalone Python aiohttp service, and it isn't proxy-routed like artifacts are (fixed port 8000, not `$PORT`).
- _Populate the rest as you build (3-5 bullets)._

## Product

- Crypto trading signal bot: analyzes live Binance market data (BTC, ETH, SOL, BNB, XRP, DOGE) across 10 confluence strategies (trend, support/resistance, patterns, smart money concepts, orderflow, etc.) and serves a live-updating charting dashboard with LONG/SHORT signals and trade plans (entry/stop/take-profit). Optional AI trade analyst (via Groq) manages one tracked call per symbol end-to-end instead of firing one-off signals.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

_Populate as you build — sharp edges, "always run X before Y" rules._

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
