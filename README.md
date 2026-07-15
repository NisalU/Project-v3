# AI Trading Signal Bot (Termux + Web Dashboard)

A pure-Python crypto trading signal bot that analyzes Binance market data with
10 strategies and serves a live charting dashboard on your phone's local IP.

Built on **aiohttp + WebSocket** for realtime updates: live trade ticks move
the last candle and the price with smooth animation, klines stream in as they
form, and analysis snapshots / signals push instantly to every connected
browser. No numpy / pandas — only `aiohttp`, `websockets` and
`requests`, so it installs cleanly on Termux.

## Strategies (confluence scored)

| Strategy | What it detects |
| --- | --- |
| EMA 7/25/99 | Trend stack alignment, fresh 7/25 crosses |
| Support / Resistance | Clustered swing levels, bounces, breakouts |
| Trendlines | Regression-fit trendlines, bounces and breaks |
| Chart Patterns | Engulfing, pin bars, double top/bottom, head & shoulders |
| Fibonacci | Retracement of dominant swing (0.5 / 0.618 golden zone) |
| Smart Money Concepts | BOS / CHoCH, order blocks, fair value gaps |
| Liquidity Sweeps | Stop hunts through equal highs/lows that reverse |
| Orderflow / CVD | Delta pressure, CVD divergence, absorption |
| Auction Market | Volume profile POC, value area acceptance/rejection |
| Fundamentals | Funding rate, open interest, long/short ratio |

Each strategy votes -1..+1 and is weighted (see `config.py`). A LONG/SHORT
signal fires when the composite score crosses the threshold (default 45/100),
together with an ATR-based trade plan (entry / stop / TP1 / TP2).

## Install on Termux

```bash
pkg update && pkg install python git
git clone <your-repo-url> signal-bot && cd signal-bot
pip install -r requirements.txt
python server.py
```

The console prints your local network URL, e.g.:

```
  Local:   http://127.0.0.1:8000
  Network: http://192.168.1.23:8000
```

Open the Network URL from any browser on the same Wi-Fi (or the Local URL on
the phone itself). To keep it running with the screen off, run
`termux-wake-lock` first.

## Configuration

Edit `config.py`:

- `SYMBOLS`, `INTERVALS` — markets offered in the dashboard dropdowns
- `WEIGHTS` — importance of each strategy in the composite score
- `SIGNAL_THRESHOLD` — how much confluence is needed to fire a signal
- `REFRESH_SECONDS` — background analysis interval
- `PORT` — web server port

Signal history persists to `signals.json`.

> Note: if `api.binance.com` is geo-restricted where you are, the bot
> automatically falls back to `data-api.binance.vision` for market data.
> Futures fundamentals (funding/OI) are skipped gracefully when unavailable.

## AI analysis layer (Groq)

On top of the 10-strategy confluence engine, `ai_analyst.py` runs a Groq-hosted
LLM acting as a **selective discretionary trader**, not a signal generator.

- It receives the full 1h confluence read, an explicit liquidity/structure
  summary (sweeps, resting liquidity pools, BOS/CHoCH events, CVD
  divergence), and a 4h higher-timeframe summary for top-down context.
- Its default answer is **WAIT**. It only calls LONG/SHORT when it can state a
  thesis, a clean location, a concrete confirmation, a logical invalidation,
  and a reward/risk of at least 1.8 — never because the engine's composite
  score or individual strategies line up.
- The bot does not just trust the model's own arithmetic: `analyze()`
  re-derives risk/reward from the actual entry/stop/tp1 numbers and checks
  entry distance in ATRs, and downgrades the call to WAIT server-side (a
  `gated: true` flag with `gate_reason`) if the model's own plan doesn't hold
  up. This mirrors the analyst's own non-negotiable rules as a backstop
  against hallucinated setups.
- Setup: set `GROQ_API_KEY` in the environment (or a local `.env` file next
  to `server.py` — handy on Termux). Without it the AI layer is disabled and
  the dashboard falls back to the raw engine signals.
- Tunables in `config.py`: `AI_INTERVAL` / `AI_HTF_INTERVAL` (chart + HTF
  context), `AI_REFRESH_SECONDS` (poll cadence), `AI_MIN_RISK_REWARD` and
  `AI_MAX_ENTRY_ATR_DISTANCE` (the server-side risk gate thresholds), and
  `GROQ_MODEL` env var to override the default model
  (`llama-3.3-70b-versatile`, with automatic fallback to other Groq models).
- `GET /api/ai?symbol=BTCUSDT` returns the latest cached call; the dashboard
  also receives live `{"type":"ai", ...}` pushes over the websocket.

## Disclaimer

Educational tool — not financial advice. Signals are algorithmic confluence
scores and discretionary AI reads, not guarantees.
# New-project
# Project-v3
