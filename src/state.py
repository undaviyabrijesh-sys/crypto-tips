from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "state" / "last_alerts.json"

def load() -> dict:
    try:
        return json.loads(PATH.read_text())
    except Exception:
        return {}

def save(state: dict) -> None:
    PATH.parent.mkdir(exist_ok=True)
    PATH.write_text(json.dumps(state, indent=2))

def already_processed(state: dict, symbol: str, candle_id: str) -> bool:
    return (state.get(symbol) or {}).get("last_candle") == candle_id

def mark_processed(state: dict, symbol: str, candle_id: str) -> None:
    state.setdefault(symbol, {})["last_candle"] = candle_id

def in_cooldown(state: dict, symbol: str, direction: str, hours: float) -> bool:
    rec = state.get(symbol)
    if not rec or rec.get("direction") != direction:
        return False
    sent = datetime.fromisoformat(rec["time"])
    return datetime.now(timezone.utc) - sent < timedelta(hours=hours)

def _close_trade(state: dict, symbol: str, trade: dict, exit_time, exit_px: float, result: str):
    entry = float(trade["entry"])
    pnl_pct = ((exit_px - entry) / entry * 100 if trade["direction"] == "LONG"
               else (entry - exit_px) / entry * 100)
    risk = abs(entry - float(trade["sl"]))
    pnl_r = (((exit_px - entry) if trade["direction"] == "LONG" else (entry - exit_px)) / risk
             if risk > 0 else 0.0)
    cost_pct = float(trade.get("round_trip_cost_pct", 0.0))
    net_pnl_pct = pnl_pct - cost_pct
    net_r = pnl_r - (cost_pct / (risk / entry * 100)) if risk > 0 else pnl_r
    row = {**trade, "exit_time": str(exit_time), "exit": exit_px, "outcome": result,
           "gross_pnl_pct": round(pnl_pct, 5), "gross_R": round(pnl_r, 4),
           "estimated_net_pnl_pct": round(net_pnl_pct, 5), "estimated_net_R": round(net_r, 4)}
    log = state.setdefault("_paper_log", [])
    log.append(row)
    del log[:-1000]
    state[symbol].pop("open_signal", None)

def update_open_signal(state: dict, symbol: str, candles, horizon_bars: int):
    rec = state.get(symbol) or {}
    trade = rec.get("open_signal")
    if not trade:
        return None
    opened_at = datetime.fromisoformat(trade["candle_time"])
    future = candles[candles.index > opened_at]
    if future.empty:
        return "OPEN"
    direction = trade["direction"]
    tp, sl = float(trade["tp"]), float(trade["sl"])
    for bar_time, bar in future.iloc[:horizon_bars].iterrows():
        op, hi, lo, cl = (float(bar[k]) for k in ("open", "high", "low", "close"))
        if direction == "LONG":
            if op <= sl: exit_px, result = op, "SL"
            elif op >= tp: exit_px, result = op, "TP"
            elif lo <= sl: exit_px, result = sl, "SL"
            elif hi >= tp: exit_px, result = tp, "TP"
            else: continue
        else:
            if op >= sl: exit_px, result = op, "SL"
            elif op <= tp: exit_px, result = op, "TP"
            elif hi >= sl: exit_px, result = sl, "SL"
            elif lo <= tp: exit_px, result = tp, "TP"
            else: continue
        _close_trade(state, symbol, trade, bar_time, exit_px, result)
        return result
    if len(future) >= horizon_bars:
        last = future.iloc[horizon_bars - 1]
        _close_trade(state, symbol, trade, future.index[horizon_bars - 1], float(last["close"]), "TIME")
        return "TIME"
    return "OPEN"

def record(state: dict, symbol: str, direction: str, candle_id: str,
           entry: float, tp: float, sl: float, horizon_bars: int, atr_value: float, round_trip_cost_pct: float = 0.158):
    rec = state.setdefault(symbol, {})
    rec["direction"] = direction
    rec["time"] = datetime.now(timezone.utc).isoformat()
    rec["last_candle"] = candle_id
    rec["open_signal"] = {
        "symbol": symbol, "direction": direction, "entry": float(entry),
        "tp": float(tp), "sl": float(sl), "candle_time": candle_id,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "horizon_bars": int(horizon_bars), "atr": float(atr_value),
        "round_trip_cost_pct": float(round_trip_cost_pct),
    }
