"""Binance market data fetcher with endpoint fallback.

Pure-Python (only `requests`) so it installs cleanly on Termux.
Thread-safe: one session per worker thread (requests.Session is not
thread-safe) and TTL caches so concurrent analyses don't hammer the API
(24h ticker and futures stats change slowly).
"""
import threading
import time

import requests

import config

_tls = threading.local()   # one requests.Session per worker thread

_spot_base = None          # cached working spot endpoint
_fut_base = None           # cached working futures endpoint
_fut_disabled_until = 0

_cache_lock = threading.Lock()
_ticker_cache = {}         # symbol -> (expires_at, data)
_futures_cache = {}        # symbol -> (expires_at, data)
TICKER_TTL = 10            # s — 24h ticker doesn't need per-snapshot fetches
FUTURES_TTL = 120          # s — funding/OI/LS move slowly; saves 3 HTTP calls per snapshot


class DataError(Exception):
    pass


def _session():
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "signal-bot/1.0"})
        _tls.session = s
    return s


def _get(base_candidates, cached, path, params):
    """Try each base URL until one responds. Returns (json, working_base)."""
    bases = ([cached] if cached else []) + [b for b in base_candidates if b != cached]
    last_err = None
    for base in bases:
        try:
            r = _session().get(base + path, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                # Binance geo-block returns 200 with {"code":0,"msg":...}
                if isinstance(data, dict) and "msg" in data and "code" in data:
                    last_err = data.get("msg")
                    continue
                return data, base
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    raise DataError(f"All endpoints failed for {path}: {last_err}")


def get_klines(symbol, interval, limit=None):
    """Return list of candle dicts (oldest -> newest)."""
    global _spot_base
    limit = limit or config.KLINE_LIMIT
    raw, _spot_base = _get(
        config.SPOT_ENDPOINTS, _spot_base, "/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )
    candles = []
    for k in raw:
        vol = float(k[5])
        taker_buy = float(k[9])
        candles.append({
            "time": k[0] // 1000,          # unix seconds (lightweight-charts format)
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": vol,
            "taker_buy": taker_buy,
            "delta": 2 * taker_buy - vol,  # taker buy - taker sell volume
        })
    return candles


def get_ticker(symbol):
    global _spot_base
    now = time.time()
    with _cache_lock:
        hit = _ticker_cache.get(symbol)
        if hit and hit[0] > now:
            return hit[1]
    data, _spot_base = _get(
        config.SPOT_ENDPOINTS, _spot_base, "/api/v3/ticker/24hr", {"symbol": symbol}
    )
    out = {
        "last": float(data["lastPrice"]),
        "change_pct": float(data["priceChangePercent"]),
        "high": float(data["highPrice"]),
        "low": float(data["lowPrice"]),
        "volume": float(data["quoteVolume"]),
    }
    with _cache_lock:
        _ticker_cache[symbol] = (now + TICKER_TTL, out)
    return out


def get_futures_stats(symbol):
    """Funding rate, open interest and long/short ratio from Binance futures.

    Returns None if the futures API is unreachable (e.g. geo-restricted);
    the fundamentals strategy degrades gracefully. Results are cached for
    FUTURES_TTL seconds — these metrics update hourly upstream.
    """
    global _fut_base, _fut_disabled_until
    now = time.time()
    with _cache_lock:
        hit = _futures_cache.get(symbol)
        if hit and hit[0] > now:
            return hit[1]
    if now < _fut_disabled_until:
        return None
    try:
        premium, _fut_base = _get(
            config.FUTURES_ENDPOINTS, _fut_base, "/fapi/v1/premiumIndex", {"symbol": symbol}
        )
        oi_hist, _fut_base = _get(
            config.FUTURES_ENDPOINTS, _fut_base, "/futures/data/openInterestHist",
            {"symbol": symbol, "period": "1h", "limit": 25},
        )
        ls_ratio, _fut_base = _get(
            config.FUTURES_ENDPOINTS, _fut_base, "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": "1h", "limit": 2},
        )
        oi_now = float(oi_hist[-1]["sumOpenInterest"]) if oi_hist else 0.0
        oi_prev = float(oi_hist[0]["sumOpenInterest"]) if oi_hist else 0.0
        out = {
            "funding_rate": float(premium.get("lastFundingRate", 0)),
            "mark_price": float(premium.get("markPrice", 0)),
            "open_interest": oi_now,
            "oi_change_pct": ((oi_now - oi_prev) / oi_prev * 100) if oi_prev else 0.0,
            "long_short_ratio": float(ls_ratio[-1]["longShortRatio"]) if ls_ratio else 1.0,
        }
        with _cache_lock:
            _futures_cache[symbol] = (now + FUTURES_TTL, out)
        return out
    except DataError:
        # Don't hammer a blocked endpoint; retry every 10 minutes.
        _fut_disabled_until = now + 600
        return None
