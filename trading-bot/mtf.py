"""Multi-timeframe helpers for the AI analyst.

Collapses a timeframe's candles into an EMA(7/25/99) snapshot the AI can use
to judge whether price is sitting in the pullback zone (good entry) or has
already run away from the EMAs (chase — bad entry), and to read the higher
timeframe's trend bias.
"""
from strategies.helpers import ema


def ema_stack(candles):
    """Latest EMA7/25/99 read for one timeframe's candles."""
    closes = [c["close"] for c in candles]
    e7, e25, e99 = ema(closes, 7), ema(closes, 25), ema(closes, 99)
    if not e99[-1]:
        return None
    price = closes[-1]
    v7, v25, v99 = e7[-1], e25[-1], e99[-1]
    if v7 > v25 > v99:
        alignment = "bullish"
    elif v7 < v25 < v99:
        alignment = "bearish"
    else:
        alignment = "mixed"
    return {
        "ema7": round(v7, 6),
        "ema25": round(v25, 6),
        "ema99": round(v99, 6),
        "alignment": alignment,
        "price_vs_ema7_pct": round((price - v7) / v7 * 100, 3),
        "price_vs_ema25_pct": round((price - v25) / v25 * 100, 3),
        "price_above_ema99": price > v99,
    }


def condensed_view(analysis):
    """Slim summary of one timeframe's analysis for use as context (higher
    or lower timeframe) rather than the primary decision timeframe."""
    stack = ema_stack(analysis["candles"])
    top = sorted(analysis["breakdown"], key=lambda b: -abs(b["contribution"]))[:3]
    ov = analysis.get("overlays", {})
    cvd = ov.get("cvd") or []
    return {
        "interval": analysis["interval"],
        "price": round(analysis["price"], 6),
        "composite": analysis["composite"],
        "direction": analysis["direction"],
        "ema_stack": stack,
        "top_reasons": [r for b in top for r in b["reasons"][:1] if r],
        "orderflow_divergence": ov.get("divergence"),
        "cvd_recent": [round(p["value"], 2) for p in cvd[-8:]],
    }
