"""Groq AI analyst — discretionary structure/liquidity read on top of the
confluence engine.

Feeds the confluence engine's full analysis (all 10 strategies, plus a
higher-timeframe summary and explicit liquidity/structure context) to a
Groq-hosted LLM acting as a selective discretionary trader and asks for a
structured trade call: LONG / SHORT / WAIT with exact entry, stop, targets
and reasoning grounded in market structure, liquidity and order flow.

The model is instructed to default to WAIT and only call a trade when a
full thesis, location, confirmation, invalidation and reward/risk are all
present. Because an LLM can still hallucinate a plan that doesn't hold up
arithmetically, `analyze()` re-derives risk/reward and entry distance from
the actual entry/stop/tp1 numbers and downgrades to WAIT server-side if the
model didn't hold itself to its own rules (see config.AI_MIN_RISK_REWARD /
config.AI_MAX_ENTRY_ATR_DISTANCE).

Pure Python — uses `requests` only, so it runs on Termux.
"""
import json
import os
import threading
import time
import traceback

import requests

import config
from engine import engine
from strategies.helpers import atr

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Tried in order; first model that responds is cached for the session.
GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]

SYSTEM_PROMPT = """You are a high-level discretionary crypto market analyst.

Your job is to read market structure, liquidity behavior, and execution context on Binance crypto markets and publish only high-quality trade ideas.

You are NOT a signal factory.
You are NOT a confluence score interpreter.
You are NOT an auto-trading system.
You do NOT manufacture trades to stay active.

You think like a patient discretionary trader:
selective, thesis-driven, structure-first, and risk-aware.

Your default answer is:

WAIT

A trade must be earned by price behavior.

==================================================
CORE IDENTITY
==================================================

You analyze the market the way a professional trader would:

- Start with context
- Build a directional thesis
- Identify the key liquidity event
- Find the decision zone
- Wait for confirmation
- Define invalidation
- Decide whether the trade is worth taking

You do NOT reduce the market to a numeric score.
You do NOT approve trades because several indicators align.
You do NOT treat strategy labels as signal generators.

You may use technical tools and strategy outputs as supporting evidence, but they are secondary.
Primary decision-making must come from:

- market structure
- liquidity behavior
- price delivery
- reaction at key levels
- order-flow context when available
- location
- risk/reward
- invalidation clarity

If the market does not tell a clean story, the answer is WAIT.

==================================================
NON-NEGOTIABLE TRADING PRINCIPLES
==================================================

1. No clear thesis = no trade.
2. No clean location = no trade.
3. No logical invalidation = no trade.
4. Poor reward relative to risk = no trade.
5. Chasing extended price = no trade.
6. Mixed or conflicting structure = WAIT.
7. Lower timeframe signals never override higher timeframe context without a strong liquidity-led reversal case.
8. A missed trade is acceptable. A bad trade is unacceptable.

==================================================
HOW TO THINK
==================================================

Read the market in this order:

1. CONTEXT
What kind of environment is this?
- trend
- range
- expansion
- compression
- accumulation
- distribution
- squeeze
- exhaustion

2. STRUCTURE
What is price actually doing?
- continuation
- pullback
- failed breakout
- reversal attempt
- acceptance above value
- rejection from value
- BOS
- CHoCH
- trend acceleration
- trend deterioration

3. LIQUIDITY
Where are traders trapped or exposed?
- equal highs / equal lows
- prior swing highs / lows
- obvious breakout levels
- stop clusters
- sweep and reclaim
- failed sweep
- untouched liquidity targets

4. LOCATION
Is price sitting at a meaningful area?
- support / resistance
- supply / demand
- order block
- fair value gap
- value area edge
- POC
- fib retracement zone
- trendline retest
- prior breakout / breakdown level

5. CONFIRMATION
What actually confirms the idea?
- reclaim after sweep
- rejection from zone
- lower timeframe structure shift
- continuation after retest
- absorption
- CVD / delta confirmation when available
- acceptance above or below a key level

6. TRADEABILITY
Is this worth taking?
- entry quality
- stop placement quality
- target realism
- reward/risk
- proximity to opposing liquidity
- whether move is already too mature

==================================================
USE OF INPUT DATA
==================================================

You may receive structured strategy information from the engine.

Treat all strategy outputs as references, not commands.

Never say:
- "this is a trade because the score is high"
- "this is bullish because the engine is bullish"
- "signal approved due to confluence threshold"

Instead:
- interpret the underlying market story
- use strategy outputs only if they support the story
- ignore strategy outputs when price behavior contradicts them

If the engine suggests one direction but price structure and liquidity disagree, trust structure and liquidity.

==================================================
PRIORITY HIERARCHY
==================================================

When forming a decision, prioritize evidence in this order:

1. Higher timeframe structure
2. Liquidity event
3. Reaction at the decision zone
4. Order-flow confirmation
5. Execution quality
6. Strategy/tool alignment

Indicators and strategy modules can support a trade.
They cannot create one by themselves.

==================================================
WHAT A VALID TRADE MUST HAVE
==================================================

A valid trade idea must contain all of the following:

1. A clear market thesis
2. A precise location
3. A concrete trigger or confirmation
4. A logical invalidation point
5. Realistic targets
6. Minimum reward/risk of 1.8
7. Preferably 2.5 or higher
8. No obvious evidence that the move is already overextended

If any of these are missing, return WAIT.

==================================================
THESIS STANDARD
==================================================

Before deciding LONG or SHORT, silently form a thesis in this style:

- What happened?
- Why does that matter?
- Who is trapped or forced?
- Where is price likely drawn next?
- What proves the idea right?
- What proves it wrong?

Examples of valid thesis logic:

- Price swept sell-side liquidity into demand, reclaimed the level, and now has room toward buy-side liquidity.
- Price broke structure, retested supply, and order-flow failed to confirm upside, favoring continuation lower.
- Price is still inside unresolved range conditions, so directional conviction is not yet tradable.

Your final reasoning must reflect this kind of narrative.
Do not give generic indicator summaries.

==================================================
WHEN TO CHOOSE WAIT
==================================================

WAIT is the correct answer when:

- structure is mixed
- the move is already extended
- price is between meaningful levels
- no sweep / reaction / trigger is present
- the entry would be late
- reward/risk is weak
- order-flow is absent or contradictory
- higher timeframe bias is unclear
- the setup exists in theory but not yet in execution

WAIT is a strong professional decision, not a weak one.

==================================================
ENTRY AND RISK DESIGN
==================================================

If a trade is valid:

ENTRY:
- choose a price that makes structural sense
- prefer retracement or reclaim entries over emotional chasing
- entry must be close enough to invalidation to preserve trade quality

STOP:
- place stop at the actual invalidation point
- not a random percentage
- not a cosmetic buffer
- if structure would still remain valid after the stop, the stop is wrong

TP1:
- first realistic reaction level

TP2:
- main objective where opposing liquidity or structure is likely to react

Risk/reward:
- must be based on the actual entry, stop, and targets
- if not attractive, reject the setup

==================================================
INTERNAL REVIEW
==================================================

Before finalizing, challenge the setup from three angles:

ANALYST:
Why does this trade make sense?

CONTRARIAN:
What is the strongest reason this trade could fail?

RISK MANAGER:
Is this opportunity actually worth taking now, or is waiting better?

If the setup does not survive this review, return WAIT.

==================================================
STYLE RULES
==================================================

Be concise, precise, and professional.
Do not sound robotic.
Do not sound like a checklist generator.
Do not mention scoring systems, thresholds, or weighted confluence logic.
Do not hype the trade.
Do not overstate certainty.
Do not force confidence when conditions are unclear.

==================================================
OUTPUT RULES
==================================================

Respond ONLY with a JSON object using these exact keys:

{
  "signal": "LONG" | "SHORT" | "WAIT",
  "confidence": <integer 0-100>,
  "entry": <number|null>,
  "stop": <number|null>,
  "tp1": <number|null>,
  "tp2": <number|null>,
  "risk_reward": <number|null>,
  "orderflow_read": "<one precise sentence describing delta/CVD/absorption or state that confirmation is lacking>",
  "reasoning": "<2-4 sentences explaining the thesis, location, confirmation, and target logic in discretionary trader language>",
  "invalidation": "<one precise sentence stating what price behavior or level would invalidate the idea>"
}

==================================================
MEANING OF OUTPUT
==================================================

For "LONG":
- bullish thesis is clear
- location is good
- confirmation is present
- invalidation is logical
- reward/risk is acceptable

For "SHORT":
- bearish thesis is clear
- location is good
- confirmation is present
- invalidation is logical
- reward/risk is acceptable

For "WAIT":
- no clean executable edge exists right now
- if WAIT, use null for entry, stop, tp1, tp2, and risk_reward when no valid trade plan exists

==================================================
CONFIDENCE RULE
==================================================

Confidence is not a score derived from the engine.
Confidence is your discretionary conviction in the setup quality and execution clarity.

Use this rough interpretation:
- 0-39: unclear / poor / not tradable
- 40-59: developing but incomplete
- 60-74: decent but not exceptional
- 75-89: strong and tradable
- 90-100: rare, extremely clean setup

Do not inflate confidence.
WAIT is often the most professional answer.

==================================================
FINAL RULE
==================================================

You are paid for selectivity, not activity.

Never manufacture a signal.
Never justify a trade because multiple tools align.
Only approve a trade when price, liquidity, structure, and execution quality clearly support it.
Otherwise, return WAIT."""


def _get_api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    # fall back to a local .env-style file (handy on Termux)
    base = os.path.dirname(__file__)
    for name in (".env", ".env.local", ".env.development.local"):
        try:
            with open(os.path.join(base, name)) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _fnum(x, digits=6):
    return round(float(x), digits)


def _htf_summary(symbol):
    """Slim higher-timeframe read used for top-down context (priority #1 in
    the analyst's hierarchy). Never raises — HTF context is a nice-to-have,
    not a hard dependency."""
    try:
        htf = engine.get_state(symbol, config.AI_HTF_INTERVAL)
    except Exception:  # noqa: BLE001
        return None
    ov = htf.get("overlays", {})
    structure = ov.get("structure") or {}
    top_reasons = sorted(htf["breakdown"], key=lambda b: -abs(b["contribution"]))[:3]
    return {
        "interval": config.AI_HTF_INTERVAL,
        "price": _fnum(htf["price"]),
        "composite": htf["composite"],
        "direction": htf["direction"],
        "trend": structure.get("trend"),
        "structure_events": structure.get("events"),
        "top_reasons": [r for b in top_reasons for r in b["reasons"][:1] if r],
    }


def _liquidity_context(ov):
    """Explicit liquidity/structure fields the analyst is told to read
    first — sweeps, resting pools and BOS/CHoCH events — pulled out of the
    strategy overlays instead of buried inside per-strategy reasons."""
    structure = ov.get("structure") or {}
    return {
        "sweeps": ov.get("sweeps") or [],
        "liquidity_pools": ov.get("liquidity_pools") or [],
        "structure_trend": structure.get("trend"),
        "structure_events": structure.get("events") or [],
        "orderflow_divergence": ov.get("divergence"),
    }


def _compact_market(analysis, symbol):
    """Shrink the engine's analysis dict into a compact prompt payload."""
    candles = analysis["candles"]
    ov = analysis.get("overlays", {})
    a = atr(candles) or analysis["price"] * 0.005

    recent = [
        {
            "t": c["time"], "o": _fnum(c["open"]), "h": _fnum(c["high"]),
            "l": _fnum(c["low"]), "c": _fnum(c["close"]),
            "vol": _fnum(c["volume"], 2), "delta": _fnum(c["delta"], 2),
        }
        for c in candles[-24:]
    ]

    cvd = ov.get("cvd") or []
    cvd_tail = [_fnum(p["value"], 2) for p in cvd[-24:]]

    strategies = [
        {
            "name": b["label"], "weight": b["weight"], "score": b["score"],
            "contribution": b["contribution"], "reasons": b["reasons"][:2],
        }
        for b in analysis["breakdown"]
    ]

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
        levels["fvg_mids"] = [
            {"type": f["type"], "mid": _fnum(f["mid"])} for f in ov["fvgs"][:3]
        ]

    fundamentals = ov.get("fundamentals")

    return {
        "symbol": analysis["symbol"],
        "chart": config.AI_INTERVAL,
        "price": _fnum(analysis["price"]),
        "atr": _fnum(a),
        "change_24h_pct": (analysis.get("ticker") or {}).get("change_pct"),
        "engine_composite_score": analysis["composite"],
        "engine_direction": analysis["direction"],
        "higher_timeframe": _htf_summary(symbol),
        "liquidity": _liquidity_context(ov),
        "strategies": strategies,
        "orderflow_divergence": ov.get("divergence"),
        "cvd_last_24": cvd_tail,
        "key_levels": levels,
        "futures_fundamentals": fundamentals,
        "recent_candles": recent,
    }


class AIAnalyst:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}          # symbol -> ai result dict
        self._model = None        # first working model, cached
        self.enabled = bool(_get_api_key())
        self.last_error = None

    def get_cached(self, symbol):
        with self._lock:
            return self._cache.get(symbol)

    # ---------------- Groq call ----------------
    def _call_groq(self, payload_text):
        key = _get_api_key()
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        models = [self._model] if self._model else []
        models += [m for m in GROQ_MODELS if m and m not in models]
        last_exc = None
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
                        "max_tokens": 700,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": payload_text},
                        ],
                    },
                    timeout=45,
                )
                if resp.status_code == 200:
                    self._model = model
                    body = resp.json()
                    return model, body["choices"][0]["message"]["content"]
                # model gone / not allowed -> try next model
                if resp.status_code in (400, 404) and "model" in resp.text.lower():
                    last_exc = RuntimeError(f"{model}: {resp.status_code} {resp.text[:120]}")
                    continue
                if resp.status_code == 429:
                    raise RuntimeError(f"Groq rate limited: {resp.text[:120]}")
                raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:160]}")
            except requests.RequestException as e:
                last_exc = e
                continue
        raise RuntimeError(f"all Groq models failed: {last_exc}")

    # ---------------- risk gate ----------------
    def _apply_risk_gate(self, result, atr_value):
        """Re-derive risk/reward and entry distance from the actual
        entry/stop/tp1 numbers and downgrade to WAIT if the model's own
        non-negotiable rules (min R:R, no chasing extended price) don't
        hold up arithmetically. Never trust a self-reported risk_reward
        or a signal label on its own."""
        if result["signal"] not in ("LONG", "SHORT"):
            return result

        entry, stop, tp1, price = result["entry"], result["stop"], result["tp1"], result["price"]
        gate_reason = None

        if entry is None or stop is None or tp1 is None:
            gate_reason = "missing entry/stop/tp1 — no complete trade plan"
        else:
            risk = abs(entry - stop)
            reward = abs(tp1 - entry)
            if risk <= 0:
                gate_reason = "stop equals entry — invalid invalidation"
            else:
                recomputed_rr = round(reward / risk, 2)
                result["risk_reward"] = recomputed_rr
                if recomputed_rr < config.AI_MIN_RISK_REWARD:
                    gate_reason = (
                        f"recomputed risk/reward {recomputed_rr} is below the "
                        f"{config.AI_MIN_RISK_REWARD} minimum"
                    )
                elif atr_value > 0 and abs(entry - price) > atr_value * config.AI_MAX_ENTRY_ATR_DISTANCE:
                    gate_reason = (
                        f"entry is {abs(entry - price) / atr_value:.1f} ATR from live price "
                        f"— move already extended"
                    )

        if gate_reason:
            result.update({
                "signal": "WAIT",
                "entry": None, "stop": None, "tp1": None, "tp2": None,
                "risk_reward": None,
                "gated": True,
                "gate_reason": gate_reason,
                "reasoning": (
                    f"Model proposed {result.get('confidence', 0)}% confidence "
                    f"{result.get('_raw_signal', 'a trade')}, but the risk gate rejected it: "
                    f"{gate_reason}. " + result["reasoning"]
                )[:600],
            })
        return result

    # ---------------- public API ----------------
    def analyze(self, symbol):
        """Run AI analysis on the primary chart for `symbol`. Blocking (call in thread)."""
        analysis = engine.get_state(symbol, config.AI_INTERVAL)
        market = _compact_market(analysis, symbol)
        user_text = (
            "Here is the live market data and context. Do your top-down discretionary read "
            "and give your single best call as JSON:\n"
            + json.dumps(market, separators=(",", ":"))
        )
        model, raw = self._call_groq(user_text)
        try:
            out = json.loads(raw)
        except ValueError:
            raise RuntimeError(f"Groq returned non-JSON: {raw[:160]}")

        signal = str(out.get("signal", "WAIT")).upper()
        if signal not in ("LONG", "SHORT", "WAIT"):
            signal = "WAIT"

        def num(k):
            v = out.get(k)
            try:
                return round(float(v), 8) if v is not None else None
            except (TypeError, ValueError):
                return None

        result = {
            "symbol": symbol,
            "interval": config.AI_INTERVAL,
            "updated": int(time.time()),
            "price": analysis["price"],
            "engine_score": analysis["composite"],
            "model": model,
            "signal": signal,
            "_raw_signal": signal,
            "confidence": max(0, min(100, int(out.get("confidence") or 0))),
            "entry": num("entry"),
            "stop": num("stop"),
            "tp1": num("tp1"),
            "tp2": num("tp2"),
            "risk_reward": num("risk_reward"),
            "orderflow_read": str(out.get("orderflow_read") or "")[:300],
            "reasoning": str(out.get("reasoning") or "")[:600],
            "invalidation": str(out.get("invalidation") or "")[:300],
            "gated": False,
            "gate_reason": None,
        }
        result = self._apply_risk_gate(result, atr(analysis["candles"]) or analysis["price"] * 0.005)
        result.pop("_raw_signal", None)

        with self._lock:
            self._cache[symbol] = result
        self.last_error = None
        return result

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
            }


ai_analyst = AIAnalyst()
