from __future__ import annotations
import argparse
import os
from datetime import datetime, timezone
import yaml

from src import state as st
from src.data import fetch_candles, interval_minutes, is_stale
from src.delta import fetch_current_price
from src.notify import send_signal, send_text
from src.signals import atr, evaluate

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-discord", action="store_true")
    args = ap.parse_args()
    try:
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
    fc = None
    state = st.load()
    sig_cfg, f_cfg = cfg["signal"], cfg["forecast"]

    for sym in cfg["symbols"]:
        name = sym["name"]
        try:
            if sym.get("skip_weekend") and datetime.now(timezone.utc).weekday() >= 5:
                print(f"{name}: weekend (thin market) - skipped")
                continue
            hist, used = fetch_candles(sym["tickers"], interval, cfg["data"]["lookback"], sym.get("delta"))
            if is_stale(hist, interval):
                print(f"{name}: data stale - skipped")
                continue
            candle_id = hist.index[-1].isoformat()
            if st.already_processed(state, name, candle_id):
                print(f"{name}: candle {candle_id} already checked - skipped")
                continue

            # Resolve an existing trade before considering a new one. A candle
            # that closes an old trade is not also used to open another trade.
            if (state.get(name) or {}).get("open_signal"):
                outcome = st.update_open_signal(state, name, hist, f_cfg["horizon_bars"])
                print(f"{name}: previous signal status {outcome}")
                st.mark_processed(state, name, candle_id)
                st.save(state)
                continue

            if fc is None:
                from src.forecast import Forecaster
                fc = Forecaster(cfg["model"]["name"], cfg["model"]["tokenizer"],
                                cfg["model"]["max_context"], cfg["model"].get("device"))
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

            atr_value = atr(hist)
            live_price, price_symbol = fetch_current_price(sym.get("delta") or [])
            max_drift = float(sig_cfg.get("max_entry_drift_atr", 0.25)) * atr_value
            drift = abs(live_price - signal.entry)
            if drift > max_drift:
                print(f"{name}: alert skipped; current {live_price:.4f} is {drift / atr_value:.2f} ATR "
                      f"from planned entry {signal.entry:.4f}")
                st.mark_processed(state, name, candle_id)
                continue

            print(f"{name}: live {live_price:.4f} ({price_symbol}); entry drift {drift / atr_value:.2f} ATR")
            if args.dry_run:
                print(signal)
                st.mark_processed(state, name, candle_id)
            else:
                send_signal(webhook, signal, interval)
                st.record(state, name, signal.direction, candle_id, signal.entry, signal.tp,
                          signal.sl, f_cfg["horizon_bars"], atr_value,\n                          sig_cfg.get("paper_round_trip_cost_pct", 0.158))
        except Exception as e:
            print(f"{name}: ERROR {e}")
    st.save(state)

if __name__ == "__main__":
    main()
