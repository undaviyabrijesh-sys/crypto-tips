from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.delta import RES_MIN, fetch_candles, regularize
from src.signals import evaluate

OUT = Path("results")


def load_data(cfg, bt, days, log=print):
    interval = cfg["data"]["interval"]
    mins = RES_MIN[interval]
    lookback = cfg["data"]["lookback"]
    horizon = cfg["forecast"]["horizon_bars"]
    end_ts = int(time.time() // (mins * 60)) * (mins * 60)
    start_ts = end_ts - days * 86400 - (lookback + horizon + 20) * mins * 60
    data = {}
    for s in bt["symbols"]:
        for dsym in s["delta"]:
            try:
                df = fetch_candles(dsym, interval, start_ts, end_ts)
                df = df[df.index < pd.Timestamp(end_ts, unit="s")]
                df = regularize(df, mins)
                if len(df) < lookback + horizon + 10:
                    raise ValueError(f"only {len(df)} candles")
                data[s["name"]] = df
                log(f"{s['name']}: using Delta {dsym}, {len(df)} candles, {df.index[0]} -> {df.index[-1]}")
                break
            except Exception as e:
                log(f"{s['name']}: {dsym} failed ({e})")
        else:
            log(f"{s['name']}: no working Delta symbol - skipped")
    if not data:
        raise SystemExit("no data could be loaded")
    return data


def run_forecasts(fc, data, cfg, days, step_hours, max_minutes, log=print):
    mins = RES_MIN[cfg["data"]["interval"]]
    lookback = cfg["data"]["lookback"]
    f = cfg["forecast"]
    horizon = f["horizon_bars"]
    end_time = min(df.index[-1] for df in data.values())
    t_last = end_time - pd.Timedelta(minutes=mins * (horizon - 1))
    t_first = t_last - pd.Timedelta(days=days)
    grid = pd.date_range(t_first.ceil(f"{step_hours}h"), t_last, freq=f"{step_hours}h")
    steps = {name: [] for name in data}
    t0 = time.time()
    log(f"{len(grid)} time steps x {len(data)} symbols = {len(grid) * len(data)} forecasts")
    for n, t in enumerate(grid, 1):
        if (time.time() - t0) / 60 > max_minutes:
            log(f"time limit reached after {n - 1} steps - analysing what is done")
            break
        for name, df in data.items():
            i = int(df.index.get_indexer([t])[0])
            if i < lookback or i + horizon > len(df):
                continue
            hist = df.iloc[i - lookback:i]
            paths = fc.sample_paths(hist, mins, horizon, f["n_paths"], f["temperature"], f["top_p"])
            steps[name].append({
                "i": i,
                "finals": [float(p["close"].iloc[-1]) for p in paths],
                "highs": [float(p["high"].max()) for p in paths],
                "lows": [float(p["low"].min()) for p in paths],
            })
        log(f"step {n}/{len(grid)} done ({(time.time() - t0) / 60:.1f} min)")
    return steps


def simulate(direction, entry, tp, sl, fut):
    hi, lo, cl = fut["high"].to_numpy(), fut["low"].to_numpy(), fut["close"].to_numpy()
    for k in range(len(fut)):
        if direction == "LONG":
            if lo[k] <= sl:
                return sl, "SL", k
            if hi[k] >= tp:
                return tp, "TP", k
        else:
            if hi[k] >= sl:
                return sl, "SL", k
            if lo[k] <= tp:
                return tp, "TP", k
    return float(cl[-1]), "TIME", len(fut) - 1


def net_r(direction, entry, exit_px, risk, cost_pct):
    pnl = (exit_px - entry) if direction == "LONG" else (entry - exit_px)
    return pnl / risk - (cost_pct / 100 * entry) / risk


def analyze(data, steps, cfg, bt, min_move_atr, min_agreement):
    sig = cfg["signal"]
    horizon = cfg["forecast"]["horizon_bars"]
    mins = RES_MIN[cfg["data"]["interval"]]
    cooldown_bars = int(sig["cooldown_hours"] * 60 / mins)
    cost_pct = 2 * (bt["costs"]["fee_pct_per_side"] + bt["costs"]["slippage_pct_per_side"])
    trades = []
    for name, df in data.items():
        busy_until, last_i, last_dir = -1, -10**9, None
        for st in steps.get(name, []):
            i = st["i"]
            hist = df.iloc[i - 30:i][["high", "low", "close"]]
            paths = [pd.DataFrame({"close": [a], "high": [b], "low": [c]})
                     for a, b, c in zip(st["finals"], st["highs"], st["lows"])]
            signal, _ = evaluate(name, hist, paths, min_move_atr, min_agreement, sig["min_rr"], horizon)
            if signal is None:
                continue
            if last_dir == signal.direction and (i - last_i) < cooldown_bars:
                continue
            last_i, last_dir = i, signal.direction
            if i <= busy_until:
                continue
            entry = float(df["open"].iloc[i])
            d, tp, sl = signal.direction, signal.tp, signal.sl
            if not ((sl < entry < tp) if d == "LONG" else (tp < entry < sl)):
                continue
            risk, reward = abs(entry - sl), abs(tp - entry)
            fut = df.iloc[i:i + horizon]
            exit_px, reason, k = simulate(d, entry, tp, sl, fut)
            md = "SHORT" if d == "LONG" else "LONG"
            m_tp = entry - reward if d == "LONG" else entry + reward
            m_sl = entry + risk if d == "LONG" else entry - risk
            m_exit, _, _ = simulate(md, entry, m_tp, m_sl, fut)
            trades.append({
                "symbol": name, "time": df.index[i], "direction": d, "entry": entry, "tp": tp, "sl": sl,
                "exit": exit_px, "reason": reason, "bars": k + 1,
                "net_R": net_r(d, entry, exit_px, risk, cost_pct),
                "mirror_net_R": net_r(md, entry, m_exit, risk, cost_pct),
                "planned_rr": reward / risk,
            })
            busy_until = i + k
    cols = ["symbol", "time", "direction", "entry", "tp", "sl", "exit", "reason", "bars",
            "net_R", "mirror_net_R", "planned_rr"]
    return pd.DataFrame(trades, columns=cols).sort_values("time").reset_index(drop=True)


def metrics(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_%": 0.0, "avg_R": 0.0, "total_R": 0.0, "profit_factor": 0.0,
                "max_dd_R": 0.0, "random_dir_avg_R": 0.0, "TP": 0, "SL": 0, "TIME": 0}
    r = t["net_R"]
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = r.cumsum()
    return {"trades": len(t), "win_%": round((r > 0).mean() * 100, 1), "avg_R": round(r.mean(), 3),
            "total_R": round(r.sum(), 2), "profit_factor": round(wins / losses, 2) if losses > 0 else 99.0,
            "max_dd_R": round((eq.cummax() - eq).max(), 2),
            "random_dir_avg_R": round(((t["net_R"] + t["mirror_net_R"]) / 2).mean(), 3),
            "TP": int((t["reason"] == "TP").sum()), "SL": int((t["reason"] == "SL").sum()),
            "TIME": int((t["reason"] == "TIME").sum())}


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_no data_\n"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def save_cache(path, data, steps, meta):
    obj = {"meta": meta, "symbols": {}}
    for name, df in data.items():
        obj["symbols"][name] = {
            "t": [x.isoformat() for x in df.index],
            "open": df["open"].tolist(), "high": df["high"].tolist(),
            "low": df["low"].tolist(), "close": df["close"].tolist(),
            "steps": steps.get(name, []),
        }
    Path(path).write_text(json.dumps(obj))


def load_cache(path):
    obj = json.loads(Path(path).read_text())
    data, steps = {}, {}
    for name, s in obj["symbols"].items():
        data[name] = pd.DataFrame({k: s[k] for k in ("open", "high", "low", "close")},
                                  index=pd.to_datetime(s["t"]))
        steps[name] = s["steps"]
    return data, steps, obj["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--bt-config", default="backtest.yaml")
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--step-hours", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--analyze", help="re-use a saved results/cache.json instead of forecasting again")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bt = yaml.safe_load(open(args.bt_config))
    OUT.mkdir(exist_ok=True)

    if args.analyze:
        data, steps, meta = load_cache(args.analyze)
    else:
        data = load_data(cfg, bt, args.days)
        from src.forecast import Forecaster
        m = cfg["model"]
        fc = Forecaster(m["name"], m["tokenizer"], m["max_context"], m.get("device"))
        steps = run_forecasts(fc, data, cfg, args.days, args.step_hours, args.max_minutes)
        meta = {"days": args.days, "step_hours": args.step_hours}
        save_cache(OUT / "cache.json", data, steps, meta)

    sig = cfg["signal"]
    base = analyze(data, steps, cfg, bt, sig["min_move_atr"], sig["min_agreement"])
    base.to_csv(OUT / "trades.csv", index=False)

    per_symbol = [{"symbol": s, **metrics(base[base.symbol == s])} for s in data]
    per_symbol.append({"symbol": "ALL", **metrics(base)})

    grid_rows = []
    for a in bt["grid"]["min_move_atr"]:
        for g in bt["grid"]["min_agreement"]:
            t = analyze(data, steps, cfg, bt, a, g)
            grid_rows.append({"min_move_atr": a, "min_agreement": g, **metrics(t)})
    pd.DataFrame(grid_rows).to_csv(OUT / "grid.csv", index=False)

    n_fc = sum(len(v) for v in steps.values())
    cost = 2 * (bt["costs"]["fee_pct_per_side"] + bt["costs"]["slippage_pct_per_side"])
    md = [
        "# Kronos backtest (Delta Exchange India candles)\n",
        f"- Forecast points: **{n_fc}** (every {meta['step_hours']}h over ~{meta['days']} days), "
        f"{cfg['forecast']['n_paths']} sampled paths each, horizon {cfg['forecast']['horizon_bars']} candles",
        f"- Entry at the open of the candle after the signal; same-candle TP+SL counts as a loss; "
        f"max 1 trade per symbol; cost {cost:.2f}% per round trip",
        f"- Live settings used: min_move_atr={sig['min_move_atr']}, min_agreement={sig['min_agreement']}, "
        f"min_rr={sig['min_rr']}\n",
        "## Results with live settings\n", md_table(per_symbol),
        "`random_dir_avg_R` = average R if the same trades had a random direction. "
        "The real `avg_R` must beat it clearly.\n",
        "## Settings comparison (all symbols)\n", md_table(grid_rows),
        "> Few trades = noisy result. Don't just pick the best row of this table: that is overfitting "
        "to a short window. Use it to see whether results are stable across neighbouring settings.\n",
    ]
    (OUT / "summary.md").write_text("\n".join(md))
    print("\n".join(md))

    if not base.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for s in data:
            t = base[base.symbol == s]
            if not t.empty:
                ax.plot(t["time"], t["net_R"].cumsum(), label=s, alpha=0.7)
        ax.plot(base["time"], base["net_R"].cumsum(), label="ALL", color="black", lw=2)
        ax.set_ylabel("cumulative R (after costs)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.savefig(OUT / "equity.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
