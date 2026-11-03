from __future__ import annotations
import argparse
import os
from datetime import datetime, timezone

import yaml

from src import state as st
from src.data import fetch_candles, interval_minutes, is_stale
from src.notify import send_signal, send_text
from src.signals import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print signals, don't send to Discord")
    ap.add_argument("--test-discord", action="store_true", help="send a test message and exit")
    args = ap.parse_args()

    try:  # optional local .env
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if args.test_discord:
        send_text(webhook, "✅ Kronos alerts: webhook works.")
        print("test message sent")
        return

    cfg = yaml.safe_load(open(args.config))
    interval = cfg["data"]["interval"]
    mins = interval_minutes(interval)

    from src.forecast import Forecaster  # heavy import (torch) only when needed
    fc = Forecaster(cfg["model"]["name"], cfg["model"]["tokenizer"],
                    cfg["model"]["max_context"], cfg["model"].get("device"))

    state = st.load()
    sig_cfg, f_cfg = cfg["signal"], cfg["forecast"]

    for sym in cfg["symbols"]:
        name = sym["name"]
        try:
            if sym.get("skip_weekend") and datetime.now(timezone.utc).weekday() >= 5:
                print(f"{name}: weekend (thin market) - skipped")
                continue
            hist, used = fetch_candles(sym["tickers"], interval, cfg["data"]["lookback"],
                                       sym.get("delta"))
            if is_stale(hist, interval):
                print(f"{name}: data stale (market closed?) - skipped")
                continue

            # The workflow may run every 5 minutes, but only forecast once per
            # newly completed 15-minute candle.
            candle_id = hist.index[-1].isoformat()
            if st.already_processed(state, name, candle_id):
                print(f"{name}: candle {candle_id} already checked - skipped")
                continue

            paths = fc.sample_paths(hist, mins, f_cfg["horizon_bars"], f_cfg["n_paths"],
                                    f_cfg["temperature"], f_cfg["top_p"])
            signal, summary = evaluate(name, hist, paths, sig_cfg["min_move_atr"],
                                       sig_cfg["min_agreement"], sig_cfg["min_rr"],
                                       f_cfg["horizon_bars"], sig_cfg.get("sl_min_atr", 0.5),
                                       sig_cfg.get("tp_rr"))
            print(f"[{used}] {summary}")
            if not signal:
                st.mark_processed(state, name, candle_id)
                continue
            if st.in_cooldown(state, name, signal.direction, sig_cfg["cooldown_hours"]):
                print(f"{name}: {signal.direction} in cooldown - not re-sent")
                st.mark_processed(state, name, candle_id)
                continue
            if args.dry_run:
                print(signal)
                st.mark_processed(state, name, candle_id)
            else:
                send_signal(webhook, signal, interval)
                st.record(state, name, signal.direction, candle_id)
        except Exception as e:
            print(f"{name}: ERROR {e}")

    st.save(state)


if __name__ == "__main__":
    main()
