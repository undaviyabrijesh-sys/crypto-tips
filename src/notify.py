from __future__ import annotations
from datetime import datetime, timezone
import requests
from .signals import Signal


def _fmt(x: float) -> str:
    return f"{x:,.2f}" if x >= 100 else f"{x:,.4f}"


def send_signal(webhook_url: str, s: Signal, interval: str) -> None:
    long = s.direction == "LONG"
    embed = {
        "title": f"{'🟢' if long else '🔴'} {s.symbol} — {s.direction} setup",
        "description": (f"Kronos-base sampled **{s.n_paths}** futures for the next "
                        f"**{s.horizon_bars} × {interval}** candles."),
        "color": 0x2ECC71 if long else 0xE74C3C,
        "fields": [
            {"name": "Entry (last close)", "value": _fmt(s.entry), "inline": True},
            {"name": "Take profit", "value": _fmt(s.tp), "inline": True},
            {"name": "Stop loss", "value": _fmt(s.sl), "inline": True},
            {"name": "Reward : Risk", "value": f"{s.rr:.2f}", "inline": True},
            {"name": "Avg forecast move", "value": f"{s.exp_move_pct:+.2f}%", "inline": True},
            {"name": "Path agreement", "value": f"{s.agreement:.0%}", "inline": True},
        ],
        "footer": {"text": "Model forecast, not financial advice. Verify on your chart."},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    r = requests.post(webhook_url, json={"embeds": [embed]}, timeout=15)
    r.raise_for_status()


def send_text(webhook_url: str, text: str) -> None:
    requests.post(webhook_url, json={"content": text}, timeout=15).raise_for_status()
