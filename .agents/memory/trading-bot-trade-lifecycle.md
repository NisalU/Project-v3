---
name: Trading bot tracked-trade lifecycle
description: The AI analyst manages one stateful trade per symbol instead of firing a fresh signal every refresh cycle — the contract to preserve when touching ai_analyst.py or trade_tracker.py.
---

The bot does not call the AI on a stateless, one-shot basis. It uses a stateful design:

- `trade_tracker.py` owns a per-symbol state machine: ARMED (limit order waiting for fill) → OPEN
  (filled, being managed) → CLOSED_WIN / CLOSED_LOSS / CLOSED_BE / INVALIDATED / EXPIRED.
- Fills, TP1/TP2 hits, stop hits, and breakeven-trailing-after-TP1 are all decided by
  `tracker.update_price()` on every live price tick (hooked into `stream.py`'s aggTrade handler) —
  no AI call needed for those transitions.
- `ai_analyst.py` checks tracker state before calling Groq: if ARMED/OPEN it runs a "manage" prompt
  (HOLD/TIGHTEN_STOP/CLOSE_NOW/INVALIDATED) against the existing call; only when idle/closed does it
  run a "prospect" prompt that can arm a brand-new call. `tracker.open_call()` itself also refuses a
  second call while one is active, as a code-level guardrail independent of prompt compliance.

**Why:** this makes the system track and manage a call end-to-end instead of forgetting it the
moment it's issued — avoids spammy "signal by signal" behavior with no follow-through.

**How to apply:** any change to `ai_analyst.py`'s call flow must preserve the ARMED/OPEN branch
routing to `_manage()` rather than `_prospect()`, or the one-active-call-per-symbol guarantee breaks
and spammy one-shot signal behavior comes back.
