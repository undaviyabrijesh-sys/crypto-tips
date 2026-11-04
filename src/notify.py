from __future__ import annotations
from datetime import datetime, timezone
import requests
from .signals import Signal


def _fmt(x: float) -> str:
    return f"{x:,.2f}" if x >= 100 else f"{x:,.4f}"


def send_signal(webhook_url: str, s: Signal, interval: str, signal_id: str) -> None:
    long = s.direction == "LONG"
    embed = {
        "title": f"{'🟢' if long else '🔴'} {s.symbol} — {s.direction} setup",
        "description": (f"Kronos-base sampled **{s.n_paths}** futures for the next "
                        f"**{s.horizon_bars} × {interval}** candles."),
        "color": 0x2ECC71 if long else 0xE74C3C,
        "fields": [
            {"name": "Signal ID", "value": signal_id, "inline": True},
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


def send_outcome(webhook_url: str, trade: dict) -> None:
    outcome = trade["outcome"]
    icon = "🟢" if outcome == "TP" else ("🔴" if outcome == "SL" else "⚪")
    label = {"TP": "TAKE PROFIT HIT", "SL": "STOP LOSS HIT", "TIME": "HORIZON EXPIRED"}[outcome]
    direction = trade["direction"]
    pnl = float(trade.get("gross_pnl_pct", 0.0))
    net = float(trade.get("estimated_net_pnl_pct", pnl))
    message = (
        f"{icon} **KRONOS — SIGNAL CLOSED: {label}**\n"
        f"Signal ID: `{trade.get('signal_id', 'legacy')}`\n"
        f"{trade['symbol']} {direction}\n"
        f"Entry: {_fmt(float(trade['entry']))} | Exit: {_fmt(float(trade['exit']))}\n"
        f"TP: {_fmt(float(trade['tp']))} | SL: {_fmt(float(trade['sl']))}\n"
        f"Gross P&L: {pnl:+.3f}% ({float(trade.get('gross_R', 0.0)):+.2f}R)\n"
        f"Estimated net P&L: {net:+.3f}% | Status: **{outcome}**"
    )
    send_text(webhook_url, message)
