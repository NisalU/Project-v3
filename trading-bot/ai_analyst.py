"""Groq AI analyst — multi-timeframe, order-flow-first trade calls that are
tracked and managed, not fired one after another like a robot.

Flow per symbol:
  1. If a call is already ARMED or OPEN (see `trade_tracker`), this is a
     MANAGEMENT check: the AI looks at fresh 4h/1h/15m data and the existing
     call and decides HOLD / TIGHTEN_STOP / CLOSE_NOW / INVALIDATED. It never
     opens a second, disconnected call while one is live.
  2. Otherwise this is a PROSPECT check: the AI does a top-down multi-
     timeframe read (4h bias -> 1h structure -> 1m order-flow timing) and
     either arms a new 1h trade call or says WAIT.

Fills, TP1/TP2, stop hits and breakeven trailing are handled by
`trade_tracker` on every live price tick — no AI call is needed for those,
which is what stops this from being a "signal by signal" spam bot.

Pure Python — uses `requests` only, so it runs on Termux.
"""
import json
import os
import re
import threading
import time
import traceback

import requests

import config
from engine import engine
from mtf import condensed_view, ema_stack
from trade_tracker import tracker

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Tried in order; first model that responds is cached for the session.
GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]

# Fallback provider when Groq is rate limited — OpenRouter's free-tier models.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# OpenRouter's free-tier catalog changes over time and slugs get deprecated —
# keep several candidates and fall through on 400/404 ("model unavailable"),
# not just 429, since a stale/renamed slug returns 404 too.
OPENROUTER_MODELS = [
    os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

PROSPECT_SYSTEM_PROMPT = """You are a professional discretionary crypto trader with 15 years of \
screen time. You do a top-down MULTI-TIMEFRAME read exactly like a human prop-desk veteran to \
decide whether to arm ONE new trade call on the 1h chart. You are given 4h (higher timeframe), \
1h (decision timeframe) and 1m (final order-flow confirmation) data, each with its own confluence of the 10 \
strategies (trend, S/R, trendlines, patterns, fibonacci, SMC, liquidity sweeps, orderflow/CVD, \
auction/volume-profile, fundamentals) plus an explicit EMA 7/25/99 stack.

1. HIGHER-TIMEFRAME BIAS (4h) — trend direction and EMA7/25/99 stack. Only take 1h setups that \
align with this bias; a counter-trend setup needs overwhelming confluence to even be considered.
2. STRUCTURE (1h) — BOS/CHoCH, key levels (support/resistance, POC/VAH/VAL, order blocks, FVGs, \
fib golden zone). Confluence of 2+ levels makes the zone A-grade.
3. ENTRY TECHNIQUE — the entry MUST be a pullback into the EMA7/EMA25 zone (or a swept liquidity \
level / order block / FVG mid that lines up with that EMA zone), in the direction of the 4h bias. \
Never chase price that has already run away from the EMAs — if price is far from EMA7/EMA25 (see \
price_vs_ema7_pct / price_vs_ema25_pct), wait for the pullback instead of calling a market chase.
4. ORDER FLOW CONFIRMATION — use 1h delta/CVD for the main read and 1m delta/CVD as the FINAL \
confirmation right before entry. You need to see delta/CVD support or absorption AT the EMA/level \
zone (the pullback should be slowing down/absorbing, not still trending through it) on the 1m tape. \
No 1m order-flow confirmation at the level = no trade.
5. CONTEXT — funding, open interest, long/short positioning as contrarian filters against crowded \
trades.

Be patient and ruthless: most of the time the correct answer is WAIT. Only call LONG or SHORT for \
an A+ setup — 4h-aligned, EMA-pullback entry, order-flow-confirmed at that level, at least 1.5 R:R \
to TP1. The entry must be one precise limit price AT the EMA/level zone, within ~2% of current \
price. The stop goes beyond the invalidating structure, not at a round number.

Respond ONLY with a JSON object with these exact keys:
{"signal": "LONG"|"SHORT"|"WAIT",
 "confidence": <int 0-100>,
 "entry": <number|null>,
 "stop": <number|null>,
 "tp1": <number|null>,
 "tp2": <number|null>,
 "risk_reward": <number|null>,
 "orderflow_read": "<one sentence: what 1h+15m delta/CVD/absorption is saying>",
 "reasoning": "<2-3 sentences: 4h bias, the EMA pullback zone, which levels line up>",
 "invalidation": "<one sentence: what would flip or cancel this idea>"}
For WAIT, entry/stop/tp1/tp2/risk_reward may be null. Numbers must be plain (no strings)."""

MANAGE_SYSTEM_PROMPT = """You are a professional discretionary crypto trader MANAGING a trade \
call you already made on the 1h chart — you are not searching for a new setup. You are given the \
existing call (its entry/stop/TP1/TP2, status and the notes logged since it was opened) plus fresh \
4h/1h/1m data (EMA stacks, order flow, structure). Decide what to do with THIS call only:

- HOLD — thesis still intact; structure and order flow still support it.
- TIGHTEN_STOP — move the stop to reduce risk (e.g. to breakeven, or under/over a fresh swing) — \
give the exact new stop price.
- CLOSE_NOW — order flow or structure has turned against the trade before hitting the stop; exit \
now to limit the loss or protect the profit made so far.
- INVALIDATED — the original thesis is broken (structure shift against you, EMA stack flipped, or \
the key level failed) even though price hasn't hit the stop yet; the trade should be abandoned.

Respond ONLY with a JSON object with these exact keys:
{"action": "HOLD"|"TIGHTEN_STOP"|"CLOSE_NOW"|"INVALIDATED",
 "new_stop": <number|null>,
 "note": "<one or two sentences, grounded in order flow / EMA / structure, explaining the decision>"}
Numbers must be plain (no strings)."""


def _get_env_key(var_name):
    key = os.environ.get(var_name, "").strip()
    if key:
        return key
    # fall back to a local .env-style file (handy on Termux)
    base = os.path.dirname(__file__)
    prefix = f"{var_name}="
    for name in (".env", ".env.local", ".env.development.local"):
        try:
            with open(os.path.join(base, name)) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith(prefix):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _get_api_key():
    return _get_env_key("GROQ_API_KEY")


def _get_openrouter_key():
    return _get_env_key("OPENROUTER_API_KEY")


def _fnum(x, digits=6):
    return round(float(x), digits)


def _recent_candles(analysis, n):
    ov = analysis.get("overlays", {})
    recent = [
        {
            "t": c["time"], "o": _fnum(c["open"]), "h": _fnum(c["high"]),
            "l": _fnum(c["low"]), "c": _fnum(c["close"]),
            "vol": _fnum(c["volume"], 2), "delta": _fnum(c["delta"], 2),
        }
        for c in analysis["candles"][-n:]
    ]
    cvd = ov.get("cvd") or []
    return recent, [_fnum(p["value"], 2) for p in cvd[-n:]]


def _levels(analysis):
    ov = analysis.get("overlays", {})
    levels = {}
    if ov.get("support"):
        levels["support"] = [_fnum(lv["price"]) for lv in ov["support"][:4]]
    if ov.get("resistance"):
        levels["resistance"] = [_fnum(lv["price"]) for lv in ov["resistance"][:4]]
    if ov.get("volume_profile"):
        vp = ov["volume_profile"]
        levels["poc"] = _fnum(vp["poc"])
        levels["vah"] = _fnum(vp["vah"])
        levels["val"] = _fnum(vp["val"])
    if ov.get("order_blocks"):
        levels["order_blocks"] = [
            {"type": ob["type"], "top": _fnum(ob["top"]), "bottom": _fnum(ob["bottom"])}
            for ob in ov["order_blocks"][:3]
        ]
    if ov.get("fvgs"):
        levels["fvg_mids"] = [{"type": f["type"], "mid": _fnum(f["mid"])} for f in ov["fvgs"][:3]]
    return levels


def _compact_1h(analysis):
    """Full detail for the decision timeframe (1h)."""
    recent, cvd_tail = _recent_candles(analysis, 24)
    strategies = [
        {
            "name": b["label"], "weight": b["weight"], "score": b["score"],
            "contribution": b["contribution"], "reasons": b["reasons"][:2],
        }
        for b in analysis["breakdown"]
    ]
    return {
        "interval": "1h",
        "price": _fnum(analysis["price"]),
        "change_24h_pct": (analysis.get("ticker") or {}).get("change_pct"),
        "engine_composite_score": analysis["composite"],
        "engine_direction": analysis["direction"],
        "ema_stack": ema_stack(analysis["candles"]),
        "strategies": strategies,
        "orderflow_divergence": (analysis.get("overlays") or {}).get("divergence"),
        "cvd_last_24": cvd_tail,
        "key_levels": _levels(analysis),
        "futures_fundamentals": (analysis.get("overlays") or {}).get("fundamentals"),
        "recent_1h_candles": recent,
    }


def _multi_timeframe_market(symbol):
    """Pull 4h / 1h / 15m analyses and build the full prospecting payload."""
    htf = engine.get_state(symbol, config.AI_HTF_INTERVAL)
    signal_tf = engine.get_state(symbol, config.AI_INTERVAL)
    ltf = engine.get_state(symbol, config.AI_LTF_INTERVAL)
    return {
        "symbol": symbol,
        "higher_timeframe_bias": condensed_view(htf),
        "decision_timeframe_1h": _compact_1h(signal_tf),
        "final_orderflow_confirmation_1m": condensed_view(ltf),
    }, signal_tf


class AIAnalyst:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}          # symbol -> ai result dict
        self._model = None        # first working model, cached
        self.enabled = bool(_get_api_key() or _get_openrouter_key())
        self.last_error = None
        self.tracker = tracker
        self._cooldown_until = 0.0   # epoch seconds; skip Groq calls until then
        self._cooldown_reason = None
        self._or_model = None
        self._or_cooldown_until = 0.0
        self._or_cooldown_reason = None
        self.active_provider = None  # last provider that actually served a result

    def get_cached(self, symbol):
        with self._lock:
            return self._cache.get(symbol)

    @staticmethod
    def _retry_after_seconds(resp):
        """Parse a usable backoff duration from a 429 response."""
        header = resp.headers.get("retry-after")
        if header:
            try:
                return max(1.0, float(header))
            except ValueError:
                pass
        try:
            msg = resp.json().get("error", {}).get("message", "")
        except ValueError:
            msg = resp.text
        m = re.search(r"try again in ([\d.]+)s", msg, re.IGNORECASE)
        if m:
            try:
                return max(1.0, float(m.group(1)))
            except ValueError:
                pass
        return 30.0  # sane default when Groq doesn't tell us how long to wait

    # ---------------- Groq call ----------------
    def _call_groq(self, system_prompt, payload_text):
        key = _get_api_key()
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        now = time.time()
        if now < self._cooldown_until:
            wait = int(self._cooldown_until - now)
            raise RuntimeError(f"Groq rate limited, cooling down {wait}s more ({self._cooldown_reason})")
        models = [self._model] if self._model else []
        models += [m for m in GROQ_MODELS if m and m not in models]
        last_exc = None
        rate_limited_models = 0
        for model in models:
            try:
                resp = requests.post(
                    GROQ_URL,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "temperature": 0.2,
                        "max_tokens": 600,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": payload_text},
                        ],
                    },
                    timeout=45,
                )
                if resp.status_code == 200:
                    self._model = model
                    body = resp.json()
                    return model, body["choices"][0]["message"]["content"]
                if resp.status_code in (400, 404) and "model" in resp.text.lower():
                    last_exc = RuntimeError(f"{model}: {resp.status_code} {resp.text[:120]}")
                    continue
                if resp.status_code == 429:
                    # This model/org is rate limited — try the next model before
                    # giving up, since only one model may be exhausted.
                    rate_limited_models += 1
                    last_exc = RuntimeError(f"{model}: rate limited: {resp.text[:120]}")
                    self._cooldown_until = max(self._cooldown_until, time.time() + self._retry_after_seconds(resp))
                    self._cooldown_reason = model
                    continue
                raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:160]}")
            except requests.RequestException as e:
                last_exc = e
                continue
        if rate_limited_models == len(models):
            # Every model we tried is rate limited — respect the cooldown we
            # just set instead of hammering Groq again next cycle.
            raise RuntimeError(f"all Groq models rate limited: {last_exc}")
        raise RuntimeError(f"all Groq models failed: {last_exc}")

    # ---------------- OpenRouter call (fallback provider) ----------------
    def _call_openrouter(self, system_prompt, payload_text):
        key = _get_openrouter_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        now = time.time()
        if now < self._or_cooldown_until:
            wait = int(self._or_cooldown_until - now)
            raise RuntimeError(f"OpenRouter rate limited, cooling down {wait}s more ({self._or_cooldown_reason})")
        models = [self._or_model] if self._or_model else []
        models += [m for m in OPENROUTER_MODELS if m and m not in models]
        last_exc = None
        rate_limited_models = 0
        for model in models:
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "temperature": 0.2,
                        # Free-tier models on OpenRouter are often reasoning
                        # models that otherwise burn the whole token budget on
                        # hidden "reasoning" before ever writing the JSON
                        # answer — cap reasoning effort so content comes back.
                        "max_tokens": 1200,
                        "reasoning": {"effort": "low"},
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": payload_text},
                        ],
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    content = (body.get("choices") or [{}])[0].get("message", {}).get("content")
                    if not content:
                        # Reasoning ate the whole token budget, or the model
                        # ignored response_format — try the next candidate.
                        last_exc = RuntimeError(f"{model}: empty content (finish_reason="
                                                 f"{(body.get('choices') or [{}])[0].get('finish_reason')})")
                        continue
                    self._or_model = model
                    return model, content
                if resp.status_code in (400, 404):
                    # Model slug unavailable/renamed/deprecated on OpenRouter's
                    # free tier — try the next candidate instead of giving up.
                    last_exc = RuntimeError(f"{model}: {resp.status_code} {resp.text[:120]}")
                    continue
                if resp.status_code == 429:
                    rate_limited_models += 1
                    last_exc = RuntimeError(f"{model}: rate limited: {resp.text[:120]}")
                    self._or_cooldown_until = max(self._or_cooldown_until, time.time() + self._retry_after_seconds(resp))
                    self._or_cooldown_reason = model
                    continue
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:160]}")
            except requests.RequestException as e:
                last_exc = e
                continue
        if rate_limited_models == len(models):
            raise RuntimeError(f"all OpenRouter models rate limited: {last_exc}")
        raise RuntimeError(f"all OpenRouter models failed: {last_exc}")

    # ---------------- combined call: Groq first, OpenRouter fallback ----------------
    def _call_llm(self, system_prompt, payload_text):
        groq_exc = None
        if _get_api_key():
            try:
                model, raw = self._call_groq(system_prompt, payload_text)
                self.active_provider = "groq"
                return f"groq/{model}", raw
            except Exception as e:  # noqa: BLE001
                groq_exc = e
        if _get_openrouter_key():
            try:
                model, raw = self._call_openrouter(system_prompt, payload_text)
                self.active_provider = "openrouter"
                return f"openrouter/{model}", raw
            except Exception as or_exc:  # noqa: BLE001
                if groq_exc:
                    raise RuntimeError(f"Groq failed ({groq_exc}); OpenRouter failed ({or_exc})")
                raise
        if groq_exc:
            raise groq_exc
        raise RuntimeError("no AI provider configured (set GROQ_API_KEY or OPENROUTER_API_KEY)")

    @staticmethod
    def _num(out, k):
        v = out.get(k)
        try:
            return round(float(v), 8) if v is not None else None
        except (TypeError, ValueError):
            return None

    # ---------------- prospecting: look for a new A+ setup ----------------
    def _prospect(self, symbol):
        market, signal_tf = _multi_timeframe_market(symbol)
        user_text = (
            "Here is the live multi-timeframe market data (4h bias, 1h decision timeframe, "
            "15m entry timing). Do your top-down professional read and give your single best "
            "trade call as JSON:\n" + json.dumps(market, separators=(",", ":"))
        )
        model, raw = self._call_llm(PROSPECT_SYSTEM_PROMPT, user_text)
        try:
            out = json.loads(raw)
        except ValueError:
            raise RuntimeError(f"{model} returned non-JSON: {raw[:160]}")

        signal = str(out.get("signal", "WAIT")).upper()
        if signal not in ("LONG", "SHORT", "WAIT"):
            signal = "WAIT"

        call = {
            "signal": signal,
            "confidence": max(0, min(100, int(out.get("confidence") or 0))),
            "entry": self._num(out, "entry"),
            "stop": self._num(out, "stop"),
            "tp1": self._num(out, "tp1"),
            "tp2": self._num(out, "tp2"),
            "risk_reward": self._num(out, "risk_reward"),
            "orderflow_read": str(out.get("orderflow_read") or "")[:300],
            "reasoning": str(out.get("reasoning") or "")[:600],
            "invalidation": str(out.get("invalidation") or "")[:300],
        }

        trade = None
        if signal in ("LONG", "SHORT"):
            trade = tracker.open_call(symbol, config.AI_INTERVAL, call, source=model)

        result = {
            "symbol": symbol,
            "interval": config.AI_INTERVAL,
            "updated": int(time.time()),
            "price": signal_tf["price"],
            "engine_score": signal_tf["composite"],
            "model": model,
            "mode": "prospect",
            **call,
            "status": trade["status"] if trade else ("WAIT"),
            "trade": trade,
        }
        with self._lock:
            self._cache[symbol] = result
        self.last_error = None
        return result

    # ---------------- management: watch/adjust an existing call ----------------
    def _manage(self, symbol, trade):
        market, signal_tf = _multi_timeframe_market(symbol)
        payload = {
            "symbol": symbol,
            "existing_call": {
                "signal": trade["signal"], "status": trade["status"],
                "entry": trade["entry"], "stop": trade["stop"],
                "tp1": trade["tp1"], "tp2": trade["tp2"], "tp1_hit": trade["tp1_hit"],
                "confidence": trade["confidence"], "filled_price": trade.get("filled_price"),
                "last_price": trade.get("last_price"),
                "reasoning": trade.get("reasoning"), "invalidation": trade.get("invalidation"),
                "recent_notes": [n["text"] for n in (trade.get("notes") or [])[-4:]],
            },
            "fresh_market": market,
        }
        user_text = (
            "Here is the trade call you are managing plus fresh multi-timeframe data. "
            "Decide HOLD / TIGHTEN_STOP / CLOSE_NOW / INVALIDATED as JSON:\n"
            + json.dumps(payload, separators=(",", ":"))
        )
        model, raw = self._call_llm(MANAGE_SYSTEM_PROMPT, user_text)
        try:
            out = json.loads(raw)
        except ValueError:
            raise RuntimeError(f"{model} returned non-JSON: {raw[:160]}")

        action = str(out.get("action", "HOLD")).upper()
        if action not in ("HOLD", "TIGHTEN_STOP", "CLOSE_NOW", "INVALIDATED"):
            action = "HOLD"
        note = str(out.get("note") or "")[:400]
        new_stop = self._num(out, "new_stop")

        if action == "TIGHTEN_STOP" and new_stop is not None:
            tracker.update_stop(symbol, new_stop, note=f"AI tightened stop to {new_stop}: {note}")
        elif action == "CLOSE_NOW":
            tracker.manual_close(symbol, "CLOSED_BE" if trade.get("tp1_hit") else "CLOSED_LOSS",
                                  f"AI closed early: {note}")
        elif action == "INVALIDATED":
            tracker.manual_close(symbol, "INVALIDATED", f"AI marked invalidated: {note}")
        else:
            tracker.add_note(symbol, f"AI: {note}", management=action)

        updated_trade = tracker.get(symbol) or trade
        result = {
            "symbol": symbol,
            "interval": config.AI_INTERVAL,
            "updated": int(time.time()),
            "price": signal_tf["price"],
            "engine_score": signal_tf["composite"],
            "model": model,
            "mode": "manage",
            "signal": updated_trade["signal"],
            "confidence": updated_trade["confidence"],
            "entry": updated_trade["entry"],
            "stop": updated_trade["stop"],
            "tp1": updated_trade["tp1"],
            "tp2": updated_trade["tp2"],
            "risk_reward": updated_trade.get("risk_reward"),
            "orderflow_read": updated_trade.get("orderflow_read", ""),
            "reasoning": note or updated_trade.get("reasoning", ""),
            "invalidation": updated_trade.get("invalidation", ""),
            "status": updated_trade["status"],
            "management_action": action,
            "trade": updated_trade,
        }
        with self._lock:
            self._cache[symbol] = result
        self.last_error = None
        return result

    # ---------------- public API ----------------
    def analyze(self, symbol):
        """Blocking (call in thread). Manages an existing tracked call if
        one is ARMED/OPEN for `symbol`; otherwise prospects for a new one."""
        trade = tracker.get(symbol)
        if trade and trade["status"] in ("ARMED", "OPEN"):
            return self._manage(symbol, trade)
        return self._prospect(symbol)

    def analyze_safe(self, symbol):
        """Like analyze() but never raises; returns cached/error placeholder."""
        try:
            return self.analyze(symbol)
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)
            traceback.print_exc()
            cached = self.get_cached(symbol)
            if cached:
                return cached
            return {
                "symbol": symbol,
                "interval": config.AI_INTERVAL,
                "updated": int(time.time()),
                "error": str(e)[:200],
                "trade": tracker.get(symbol),
            }


ai_analyst = AIAnalyst()
