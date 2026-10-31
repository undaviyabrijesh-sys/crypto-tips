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


def in_cooldown(state: dict, symbol: str, direction: str, hours: float) -> bool:
    rec = state.get(symbol)
    if not rec or rec.get("direction") != direction:
        return False
    sent = datetime.fromisoformat(rec["time"])
    return datetime.now(timezone.utc) - sent < timedelta(hours=hours)


def record(state: dict, symbol: str, direction: str) -> None:
    state[symbol] = {"direction": direction, "time": datetime.now(timezone.utc).isoformat()}
