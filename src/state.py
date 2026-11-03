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
    rec = state.get(symbol) or {}
    return rec.get("last_candle") == candle_id


def mark_processed(state: dict, symbol: str, candle_id: str) -> None:
    rec = state.setdefault(symbol, {})
    rec["last_candle"] = candle_id


def in_cooldown(state: dict, symbol: str, direction: str, hours: float) -> bool:
    rec = state.get(symbol)
    if not rec or rec.get("direction") != direction:
        return False
    sent = datetime.fromisoformat(rec["time"])
    return datetime.now(timezone.utc) - sent < timedelta(hours=hours)


def record(state: dict, symbol: str, direction: str, candle_id: str | None = None) -> None:
    rec = state.setdefault(symbol, {})
    rec["direction"] = direction
    rec["time"] = datetime.now(timezone.utc).isoformat()
    if candle_id is not None:
        rec["last_candle"] = candle_id
