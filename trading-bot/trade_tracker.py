"""Signal lifecycle tracker.

This is what turns the AI analyst from a stateless "fire a new call every
cycle" robot into a system that manages ONE tracked trade per symbol from
idea to close:

    IDLE -> ARMED (limit order waiting for entry) -> OPEN (filled, managed)
         -> CLOSED_WIN / CLOSED_LOSS / CLOSED_BE / INVALIDATED / EXPIRED

While a trade is ARMED or OPEN for a symbol, the analyst is not allowed to
open a second, disconnected call for that symbol — it can only *manage* the
existing one. Price progression (fills, TP1/TP2, stop, breakeven trail) is
tracked on every tick, cheaply, with no AI call required.
"""
import json
import os
import threading
import time

import config

STATE_FILE = os.path.join(os.path.dirname(__file__), "trade_state.json")

# How long an ARMED (unfilled) call may wait for its entry price before it's
# dropped as stale.
ARM_TIMEOUT_SECONDS = getattr(config, "AI_ARM_TIMEOUT_SECONDS", 6 * 3600)

ACTIVE_STATUSES = ("ARMED", "OPEN")


class TradeTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._trades = self._load()  # symbol -> trade dict

    # ---------------- persistence ----------------
    def _load(self):
        try:
            with open(STATE_FILE) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return {}

    def _save(self):
        try:
            with open(STATE_FILE, "w") as fh:
                json.dump(self._trades, fh)
        except Exception:  # noqa: BLE001
            pass

    # ---------------- reads ----------------
    def get(self, symbol):
        with self._lock:
            t = self._trades.get(symbol)
            return dict(t) if t else None

    def is_active(self, symbol):
        t = self.get(symbol)
        return bool(t and t["status"] in ACTIVE_STATUSES)

    def active_symbols(self):
        with self._lock:
            return [s for s, t in self._trades.items() if t["status"] in ACTIVE_STATUSES]

    # ---------------- lifecycle ----------------
    def open_call(self, symbol, interval, call, source="ai"):
        """Arm a new tracked trade from an AI call. Refuses (returns the
        existing trade unchanged) if one is already ARMED/OPEN for this
        symbol — this is the guardrail against robot-style signal spam."""
        with self._lock:
            existing = self._trades.get(symbol)
            if existing and existing["status"] in ACTIVE_STATUSES:
                return dict(existing)
            now = int(time.time())
            trade = {
                "symbol": symbol,
                "interval": interval,
                "signal": call["signal"],
                "confidence": call.get("confidence"),
                "entry": call.get("entry"),
                "stop": call.get("stop"),
                "tp1": call.get("tp1"),
                "tp2": call.get("tp2"),
                "risk_reward": call.get("risk_reward"),
                "orderflow_read": call.get("orderflow_read"),
                "reasoning": call.get("reasoning"),
                "invalidation": call.get("invalidation"),
                "status": "ARMED",
                "tp1_hit": False,
                "created_at": now,
                "armed_at": now,
                "filled_at": None,
                "filled_price": None,
                "closed_at": None,
                "close_reason": None,
                "last_price": None,
                "last_management": None,
                "notes": [{
                    "t": now,
                    "text": f"Armed by {source}: {call['signal']} limit @ {call.get('entry')}, "
                            f"stop {call.get('stop')}, TP1 {call.get('tp1')}, TP2 {call.get('tp2')}",
                }],
            }
            self._trades[symbol] = trade
            self._save()
            return dict(trade)

    def add_note(self, symbol, text, management=None):
        with self._lock:
            t = self._trades.get(symbol)
            if not t:
                return
            t["notes"] = (t.get("notes") or [])[-9:] + [{"t": int(time.time()), "text": text}]
            if management:
                t["last_management"] = management
            self._save()

    def update_stop(self, symbol, new_stop, note=None):
        with self._lock:
            t = self._trades.get(symbol)
            if not t or t["status"] not in ACTIVE_STATUSES or new_stop is None:
                return
            t["stop"] = new_stop
            if note:
                t["notes"] = (t.get("notes") or [])[-9:] + [{"t": int(time.time()), "text": note}]
            self._save()

    def manual_close(self, symbol, status, reason):
        with self._lock:
            t = self._trades.get(symbol)
            if not t or t["status"] not in ACTIVE_STATUSES:
                return
            t["status"] = status
            t["close_reason"] = reason
            t["closed_at"] = int(time.time())
            self._save()

    # ---------------- price-driven state machine ----------------
    def update_price(self, symbol, price):
        """Progress ARMED/OPEN state for `symbol` given the latest traded
        price. Cheap, synchronous, called on every live tick — no AI call
        needed for fills, TP hits, stop hits or breakeven trailing.
        Returns the updated trade dict only when the status just changed
        (so callers can push a lightweight update), else None."""
        with self._lock:
            t = self._trades.get(symbol)
            if not t or t["status"] not in ACTIVE_STATUSES:
                return None
            t["last_price"] = price
            long = t["signal"] == "LONG"
            changed = False

            if t["status"] == "ARMED":
                filled = (price <= t["entry"]) if long else (price >= t["entry"])
                invalidated = (price <= t["stop"]) if long else (price >= t["stop"])
                stale = int(time.time()) - t["armed_at"] > ARM_TIMEOUT_SECONDS
                if invalidated:
                    t["status"] = "INVALIDATED"
                    t["close_reason"] = "Price hit the stop level before the entry ever filled"
                    t["closed_at"] = int(time.time())
                    changed = True
                elif filled:
                    t["status"] = "OPEN"
                    t["filled_at"] = int(time.time())
                    t["filled_price"] = price
                    t["notes"] = (t.get("notes") or [])[-9:] + [
                        {"t": int(time.time()), "text": f"Entry filled @ {price}"}]
                    changed = True
                elif stale:
                    t["status"] = "EXPIRED"
                    t["close_reason"] = "Entry never filled within the arm window"
                    t["closed_at"] = int(time.time())
                    changed = True

            elif t["status"] == "OPEN":
                hit_tp1 = (price >= t["tp1"]) if long else (price <= t["tp1"])
                hit_tp2 = (price >= t["tp2"]) if long else (price <= t["tp2"])
                hit_stop = (price <= t["stop"]) if long else (price >= t["stop"])
                if not t["tp1_hit"] and hit_tp1 and t["entry"] is not None:
                    t["tp1_hit"] = True
                    t["stop"] = t["entry"]  # trail stop to breakeven after TP1
                    t["notes"] = (t.get("notes") or [])[-9:] + [
                        {"t": int(time.time()), "text": "TP1 hit — stop trailed to breakeven"}]
                    changed = True
                if hit_tp2:
                    t["status"] = "CLOSED_WIN"
                    t["close_reason"] = "TP2 reached"
                    t["closed_at"] = int(time.time())
                    changed = True
                elif hit_stop:
                    t["status"] = "CLOSED_BE" if t["tp1_hit"] else "CLOSED_LOSS"
                    t["close_reason"] = "Stop hit" + (" at breakeven after TP1" if t["tp1_hit"] else "")
                    t["closed_at"] = int(time.time())
                    changed = True

            if changed:
                self._save()
                return dict(t)
            return None


tracker = TradeTracker()
