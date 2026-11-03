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
EDGES = [0.0, 0.625, 0.75, 0.875, 1.01]
LABELS = ["under 62%", "62-75%", "75-88%", "88-100%"]


# ---------------------------------------------------------------- data
def load_data(cfg, bt, days, log=print):
    interval = cfg["data"]["interval"]
    mins = RES_MIN[interval]
    lookback = cfg["data"]["lookback"]
    horizon = cfg["forecast"]["horizon_bars"]
    end_ts = int(time.time() // (mins * 60)) * (mins * 60)      # start of the still-forming candle
    start_ts = end_ts - int(days * 86400) - (lookback + horizon + 20) * mins * 60
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


# ---------------------------------------------------------------- forecasting
def run_forecasts(fc, data, cfg, days, step_hours, max_minutes, skip_weekend=(), steps=None,
                  checkpoint=None, log=print):
    mins = RES_MIN[cfg["data"]["interval"]]
    lookback = cfg["data"]["lookback"]
    f = cfg["forecast"]
    horizon = f["horizon_bars"]
    end_time = min(df.index[-1] for df in data.values())
    t_last = end_time - pd.Timedelta(minutes=mins * (horizon - 1))
    t_first = t_last - pd.Timedelta(days=days)
    grid = pd.date_range(t_first.ceil(f"{step_hours}h"), t_last, freq=f"{step_hours}h")
    steps = steps if steps is not None else {}
    for name in data:
        steps.setdefault(name, [])
    done = {name: {s["i"] for s in steps[name]} for name in data}
    already = sum(len(v) for v in steps.values())
    t0 = time.time()
    log(f"{len(grid)} time steps, {len(data)} symbols (already done: {already} forecasts)")
    for n, t in enumerate(grid, 1):
        if (time.time() - t0) / 60 > max_minutes:
            log(f"time limit reached after {n - 1} steps - analysing what is done")
            break
        for name, df in data.items():
            if name in skip_weekend and t.weekday() >= 5:
                continue
            i = int(df.index.get_indexer([t])[0])
            if i < lookback or i + horizon > len(df) or i in done[name]:
                continue
            hist = df.iloc[i - lookback:i]
            paths = fc.sample_paths(hist, mins, horizon, f["n_paths"], f["temperature"], f["top_p"])
            steps[name].append({
                "i": i,
                "finals": [float(p["close"].iloc[-1]) for p in paths],
                "highs": [float(p["high"].max()) for p in paths],
                "lows": [float(p["low"].min()) for p in paths],
            })
            done[name].add(i)
        if checkpoint and n % 3 == 0:
            checkpoint(steps)
        log(f"step {n}/{len(grid)} done ({(time.time() - t0) / 60:.1f} min)")
    if checkpoint:
        checkpoint(steps)
    return steps


# ---------------------------------------------------------------- does the model predict direction at all?
def direction_stats(data, steps, cfg, min_agreement):
    h = cfg["forecast"]["horizon_bars"]
    recs = []
    for name, df in data.items():
        close = df["close"].to_numpy()
        for st in steps.get(name, []):
            i = st["i"]
            if i + h - 1 >= len(close):
                continue
            ref, end = close[i - 1], close[i + h - 1]
            finals = np.array(st["finals"])
            n = len(finals)
            up = int((finals > ref).sum())
            d = 1 if finals.mean() > ref else -1
            # Match live evaluate(): agreement is the share of sampled paths
            # aligned with the mean-forecast direction, not always the majority.
            agreement = up / n if d == 1 else (n - up) / n
            recs.append((agreement, d, end / ref - 1))
    if not recs:
        return [], 50.0, "no forecasts"
    a = np.array(recs)
    agree, d, actual = a[:, 0], a[:, 1], a[:, 2]
    hit = np.sign(actual) == d

    def row(label, mask):
        n = int(mask.sum())
        if n == 0:
            return {"agreement": label, "windows": 0, "hit_%": "-", "ci95_+-": "-", "avg_move_in_dir_%": "-"}
        p = float(hit[mask].mean())
        ci = 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)
        return {"agreement": label, "windows": n, "hit_%": round(p * 100, 1), "ci95_+-": round(ci * 100, 1),
                "avg_move_in_dir_%": round(float((actual[mask] * d[mask]).mean()) * 100, 3)}

    rows = [row(lab, (agree >= lo) & (agree < hi)) for lab, lo, hi in zip(LABELS, EDGES[:-1], EDGES[1:])]
    rows.append(row("ALL windows", np.ones(len(a), bool)))
    live = agree >= min_agreement - 1e-9
    rows.append(row(f"at/above live setting ({min_agreement:.0%})", live))
    up_rate = float((actual > 0).mean()) * 100
    naive = max(up_rate, 100 - up_rate)

    n = int(live.sum())
    if n < 30:
        verdict = f"Only {n} windows at the live agreement level - too few to judge direction skill."
    else:
        p = float(hit[live].mean())
        ci = 1.96 * np.sqrt(p * (1 - p) / n)
        bar = max(0.5, naive / 100)
        if p - ci > bar:
            verdict = (f"Direction hit rate {p:.1%} +- {ci:.1%} over {n} windows beats the naive "
                       f"{naive:.0f}% bar: a possible edge (still check costs/stops and out-of-sample).")
        else:
            verdict = (f"Direction hit rate {p:.1%} +- {ci:.1%} over {n} windows does NOT clearly beat the naive "
                       f"{naive:.0f}% bar: no clear direction edge detected on this sample.")
    return rows, naive, verdict


# ---------------------------------------------------------------- trade simulation
def simulate(direction, entry, tp, sl, fut):
    """Walk candles with conservative intrabar handling and gap-aware fills.

    If a candle opens through a stop, fill at its open (worse than the stop).
    If it opens through a resting target, fill at the open (price improvement).
    If both barriers are touched intrabar, assume stop-first because OHLC data
    cannot reveal the order of touches.
    """
    op = fut["open"].to_numpy()
    hi = fut["high"].to_numpy()
    lo = fut["low"].to_numpy()
    cl = fut["close"].to_numpy()
    for k in range(len(fut)):
        if direction == "LONG":
            if op[k] <= sl:
                return float(op[k]), "SL", k
            if op[k] >= tp:
                return float(op[k]), "TP", k
            if lo[k] <= sl:
                return sl, "SL", k
            if hi[k] >= tp:
                return tp, "TP", k
        else:
            if op[k] >= sl:
                return float(op[k]), "SL", k
            if op[k] <= tp:
                return float(op[k]), "TP", k
            if hi[k] >= sl:
                return sl, "SL", k
            if lo[k] <= tp:
                return tp, "TP", k
    return float(cl[-1]), "TIME", len(fut) - 1


def net_r(direction, entry, exit_px, risk, cost_pct):
    pnl = (exit_px - entry) if direction == "LONG" else (entry - exit_px)
    return pnl / risk - (cost_pct / 100 * entry) / risk


def analyze(data, steps, cfg, bt, min_move_atr, min_agreement, sl_min_atr=None,
            exit_mode="forecast", min_rr=None, fixed_rr=None):
    """exit_mode: forecast (target from forecast paths) | fixed_rr | time_only (no TP/SL, exit at horizon)."""
    sig = cfg["signal"]
    if sl_min_atr is None:
        sl_min_atr = sig.get("sl_min_atr", 0.5)
    if min_rr is None:
        min_rr = sig["min_rr"]
    target_rr = fixed_rr if fixed_rr is not None else (sig.get("tp_rr") or bt["exits"]["fixed_rr"])
    tp_rr = target_rr if exit_mode == "fixed_rr" else None
    horizon = cfg["forecast"]["horizon_bars"]
    mins = RES_MIN[cfg["data"]["interval"]]
    cooldown_bars = int(sig["cooldown_hours"] * 60 / mins)
    cost_pct = 2 * (bt["costs"]["fee_pct_per_side"] + bt["costs"]["slippage_pct_per_side"])
    trades = []
    for name, df in data.items():
        busy_until, last_i, last_dir = -1, -10**9, None
        for st in steps.get(name, []):
            i = st["i"]
            # Use only candles strictly before the forecast timestamp (no look-ahead).
            prior = df.iloc[:i]
            hist = prior.iloc[-200:][["high", "low", "close"]]
            trend_1h = prior.resample("1h", label="left", closed="left").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum", "amount": "sum"
            }).dropna(subset=["open", "high", "low", "close"])
            trend_4h = prior.resample("4h", label="left", closed="left").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum", "amount": "sum"
            }).dropna(subset=["open", "high", "low", "close"])
            # Keep only fully closed higher-timeframe candles.
            last_base_end = prior.index[-1] + pd.Timedelta(minutes=mins)
            trend_1h = trend_1h[trend_1h.index + pd.Timedelta(hours=1) <= last_base_end]
            trend_4h = trend_4h[trend_4h.index + pd.Timedelta(hours=4) <= last_base_end]
            trend_frames = {"15m": hist, "1h": trend_1h, "4h": trend_4h}
            paths = [pd.DataFrame({"close": [a], "high": [b], "low": [c]})
                     for a, b, c in zip(st["finals"], st["highs"], st["lows"])]
            signal, _ = evaluate(name, hist, paths, min_move_atr, min_agreement, min_rr,
                                 horizon, sl_min_atr, tp_rr, trend_frames,
                                 sig.get("min_trend_score", 2))
            if signal is None:
                continue
            if i <= busy_until:
                continue                                   # one trade at a time per symbol
            if last_dir == signal.direction and (i - last_i) < cooldown_bars:
                continue
            last_i, last_dir = i, signal.direction
            entry = float(df["open"].iloc[i])              # enter at the NEXT candle open
            d = signal.direction
            tp = entry + (signal.tp - signal.entry)        # keep the planned distances,
            sl = entry + (signal.sl - signal.entry)        # measured from the real entry price
            risk, reward = abs(entry - sl), abs(tp - entry)
            fut = df.iloc[i:i + horizon]
            md = "SHORT" if d == "LONG" else "LONG"        # mirror = opposite side, same distances
            if exit_mode == "time_only":
                exit_px, reason, k = float(fut["close"].iloc[-1]), "TIME", len(fut) - 1
                m_exit = exit_px
            else:
                exit_px, reason, k = simulate(d, entry, tp, sl, fut)
                m_tp = entry - reward if d == "LONG" else entry + reward
                m_sl = entry + risk if d == "LONG" else entry - risk
                m_exit, _, _ = simulate(md, entry, m_tp, m_sl, fut)

            # Diagnostic only: after a stop, did price later reach the original
            # target before the forecast horizon ended? This helps distinguish
            # premature stops from forecasts that simply went the wrong way.
            after_stop_target = False
            if reason == "SL" and k + 1 < len(fut):
                later = fut.iloc[k + 1:]
                after_stop_target = bool(
                    (later["high"] >= tp).any() if d == "LONG" else (later["low"] <= tp).any()
                )

            # Excursion strictly before the exit candle. Excluding the exit
            # candle avoids pretending OHLC data reveals intrabar ordering.
            prior = fut.iloc[:k]
            if prior.empty:
                mfe_prior_r = mae_prior_r = 0.0
            elif d == "LONG":
                mfe_prior_r = max(0.0, (float(prior["high"].max()) - entry) / risk)
                mae_prior_r = max(0.0, (entry - float(prior["low"].min())) / risk)
            else:
                mfe_prior_r = max(0.0, (entry - float(prior["low"].min())) / risk)
                mae_prior_r = max(0.0, (float(prior["high"].max()) - entry) / risk)

            trades.append({
                "symbol": name, "time": df.index[i], "direction": d, "entry": entry, "tp": tp, "sl": sl,
                "exit": exit_px, "reason": reason, "bars": k + 1,
                "net_R": net_r(d, entry, exit_px, risk, cost_pct),
                "mirror_net_R": net_r(md, entry, m_exit, risk, cost_pct),
                "planned_rr": reward / risk, "risk_pct": risk / entry * 100,
                "mfe_prior_exit_R": mfe_prior_r, "mae_prior_exit_R": mae_prior_r,
                "after_stop_target": after_stop_target,
            })
            busy_until = i + k
    cols = ["symbol", "time", "direction", "entry", "tp", "sl", "exit", "reason", "bars",
            "net_R", "mirror_net_R", "planned_rr", "risk_pct",
            "mfe_prior_exit_R", "mae_prior_exit_R", "after_stop_target"]
    return pd.DataFrame(trades, columns=cols).sort_values("time").reset_index(drop=True)


def metrics(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_%": 0.0, "avg_R": 0.0, "total_R": 0.0, "profit_factor": 0.0,
                "max_dd_R": 0.0, "random_dir_avg_R": 0.0, "avg_stop_%": 0.0,
                "avg_mfe_prior_exit_R": 0.0, "avg_mae_prior_exit_R": 0.0,
                "stopped_then_target_%": 0.0, "TP": 0, "SL": 0, "TIME": 0}
    r = t["net_R"]
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    # Include the initial zero-equity point; otherwise an initial losing trade
    # is incorrectly treated as the first equity peak and drawdown is understated.
    eq = np.concatenate(([0.0], r.cumsum().to_numpy()))
    drawdown = np.maximum.accumulate(eq) - eq
    stopped = t[t["reason"] == "SL"]
    stopped_then_target = float(stopped["after_stop_target"].mean() * 100) if len(stopped) else 0.0
    return {"trades": len(t), "win_%": round((r > 0).mean() * 100, 1), "avg_R": round(r.mean(), 3),
            "total_R": round(r.sum(), 2), "profit_factor": round(wins / losses, 2) if losses > 0 else 99.0,
            "max_dd_R": round(float(drawdown.max()), 2),
            "random_dir_avg_R": round(((t["net_R"] + t["mirror_net_R"]) / 2).mean(), 3),
            "avg_stop_%": round(t["risk_pct"].mean(), 3),
            "avg_mfe_prior_exit_R": round(t["mfe_prior_exit_R"].mean(), 3),
            "avg_mae_prior_exit_R": round(t["mae_prior_exit_R"].mean(), 3),
            "stopped_then_target_%": round(stopped_then_target, 1),
            "TP": int((t["reason"] == "TP").sum()), "SL": int((t["reason"] == "SL").sum()),
            "TIME": int((t["reason"] == "TIME").sum())}


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_no data_\n"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- cache
def save_cache(path, data, steps, meta):
    obj = {"meta": meta, "symbols": {}}
    for name, df in data.items():
        obj["symbols"][name] = {
            "t": [x.isoformat() for x in df.index],
            "open": df["open"].tolist(), "high": df["high"].tolist(),
            "low": df["low"].tolist(), "close": df["close"].tolist(),
            "volume": df["volume"].tolist(), "amount": df["amount"].tolist(),
            "steps": steps.get(name, []),
        }
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)                                      # never leaves a half-written file


def load_cache(path):
    obj = json.loads(Path(path).read_text())
    data, steps = {}, {}
    for name, s in obj["symbols"].items():
        n = len(s["t"])
        cols = {k: s[k] for k in ("open", "high", "low", "close")}
        cols["volume"] = s.get("volume") or [0.0] * n
        cols["amount"] = s.get("amount") or [0.0] * n
        data[name] = pd.DataFrame(cols, index=pd.to_datetime(s["t"]))
        steps[name] = s["steps"]
    return data, steps, obj["meta"]


def apply_meta(cfg, meta):
    if meta.get("interval"):
        cfg["data"]["interval"] = meta["interval"]
    if meta.get("horizon"):
        cfg["forecast"]["horizon_bars"] = meta["horizon"]
    if meta.get("n_paths"):
        cfg["forecast"]["n_paths"] = meta["n_paths"]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--bt-config", default="backtest.yaml")
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--step-hours", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--analyze", help="re-use a saved results/cache.json (no forecasting)")
    ap.add_argument("--resume", help="continue forecasting from a saved results/cache.json")
    ap.add_argument("--interval", help="override candle size, e.g. 1h")
    ap.add_argument("--horizon", type=int, help="override forecast length in candles")
    ap.add_argument("--n-paths", type=int, help="override sampled paths per forecast (e.g. 16)")
    ap.add_argument("--lookback", type=int, help="override candles fed to the model")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bt = yaml.safe_load(open(args.bt_config))
    if args.interval:
        cfg["data"]["interval"] = args.interval
    if args.horizon:
        cfg["forecast"]["horizon_bars"] = args.horizon
    if args.n_paths:
        cfg["forecast"]["n_paths"] = args.n_paths
    if args.lookback:
        cfg["data"]["lookback"] = args.lookback
    OUT.mkdir(exist_ok=True)
    skip_wk = {s["name"] for s in bt["symbols"] if s.get("skip_weekend")}

    if args.analyze:
        data, steps, meta = load_cache(args.analyze)
        apply_meta(cfg, meta)
    else:
        steps = None
        if args.resume:
            data, steps, meta = load_cache(args.resume)
            apply_meta(cfg, meta)
            days, step_hours = meta["days"], meta["step_hours"]
        else:
            data = load_data(cfg, bt, args.days)
            days, step_hours = args.days, args.step_hours
        meta = {"days": days, "step_hours": step_hours, "interval": cfg["data"]["interval"],
                "horizon": cfg["forecast"]["horizon_bars"], "n_paths": cfg["forecast"]["n_paths"]}
        from src.forecast import Forecaster          # heavy import only when forecasting
        m = cfg["model"]
        fc = Forecaster(m["name"], m["tokenizer"], m["max_context"], m.get("device"))
        steps = run_forecasts(fc, data, cfg, days, step_hours, args.max_minutes, skip_wk, steps,
                              checkpoint=lambda s: save_cache(OUT / "cache.json", data, s, meta))

    sig = cfg["signal"]
    live_exit = "fixed_rr" if sig.get("tp_rr") else "forecast"
    base = analyze(data, steps, cfg, bt, sig["min_move_atr"], sig["min_agreement"], exit_mode=live_exit)
    base.to_csv(OUT / "trades.csv", index=False)
    per_symbol = [{"symbol": s, **metrics(base[base.symbol == s])} for s in data]
    per_symbol.append({"symbol": "ALL", **metrics(base)})

    dir_rows, naive, verdict = direction_stats(data, steps, cfg, sig["min_agreement"])

    exit_rows = []
    rr_grid = bt["grid"].get("fixed_rr", [bt["exits"]["fixed_rr"]])
    for mode in ("forecast", "fixed_rr", "time_only"):
        for sl_a in bt["grid"]["sl_min_atr"]:
            mode_rrs = rr_grid if mode == "fixed_rr" else [None]
            for rr_target in mode_rrs:
                t = analyze(
                    data, steps, cfg, bt, sig["min_move_atr"], sig["min_agreement"],
                    sl_a, mode, min_rr=0.0, fixed_rr=rr_target,
                )
                exit_rows.append({
                    "exit": mode, "sl_min_atr": sl_a, "fixed_rr": rr_target, **metrics(t)
                })
    move_rows = []
    for a in bt["grid"]["min_move_atr"]:
        t = analyze(data, steps, cfg, bt, a, sig["min_agreement"], exit_mode=live_exit)
        move_rows.append({"min_move_atr": a, **metrics(t)})
    pd.DataFrame(exit_rows).to_csv(OUT / "grid.csv", index=False)

    n_fc = sum(len(v) for v in steps.values())
    cost = 2 * (bt["costs"]["fee_pct_per_side"] + bt["costs"]["slippage_pct_per_side"])
    md = [
        "# Kronos backtest (Delta Exchange India candles)\n",
        f"- {cfg['data']['interval']} candles, horizon {cfg['forecast']['horizon_bars']} candles, "
        f"{cfg['forecast']['n_paths']} sampled paths; **{n_fc}** forecasts "
        f"(every {meta['step_hours']}h over ~{meta['days']} days)",
        f"- Entry at the open of the candle after the signal; same-candle TP+SL counts as a loss; "
        f"max 1 trade per symbol; cost {cost:.2f}% per round trip",
        f"- Live settings: min_move_atr={sig['min_move_atr']}, min_agreement={sig['min_agreement']}, "
        f"min_rr={sig['min_rr']}, sl_min_atr={sig.get('sl_min_atr', 0.5)}, exit={live_exit}\n",
        "## 1. Can the model predict direction at all?\n",
        f"**{verdict}**\n",
        md_table(dir_rows),
        f"`hit_%` = how often price really moved the way Kronos said over the whole horizon. "
        f"A coin flip is 50%; always guessing the market's majority direction scored {naive:.0f}% here. "
        f"`ci95_+-` is the uncertainty. Windows overlap, so the real sample is smaller than shown.\n",
        "## 2. Trades with live settings\n", md_table(per_symbol),
        "`random_dir_avg_R` = average R if the same trades had a random direction. "
        "The real `avg_R` must beat it clearly. R = multiples of the planned stop distance, after costs.\n",
        "## 3. Exit style x minimum stop size (same trades, no reward:risk filter)\n", md_table(exit_rows),
        "`forecast` = target from forecast paths, `fixed_rr` = fixed multiple of the stop, "
        "`time_only` = no stop/target, exit when the horizon ends. The fixed-RR rows compare every configured target multiple.\n",
        "`mfe_prior_exit_R` and `mae_prior_exit_R` measure favourable/adverse excursion strictly before the exit candle. "
        "`stopped_then_target_%` is the share of stop-outs after which the original target was reached later in the same forecast horizon; it is diagnostic, not a tradable result.\n",
        "## 4. Minimum forecast move (live stop and exit)\n", md_table(move_rows),
        "> Few trades = noisy. Don't pick the single best row: that is overfitting to a short window. "
        "Look for results that hold across neighbouring rows.\n",
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
