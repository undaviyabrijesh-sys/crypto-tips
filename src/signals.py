from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Signal:
    symbol: str
    direction: str
    entry: float
    tp: float
    sl: float
    rr: float
    exp_move_pct: float
    agreement: float
    horizon_bars: int
    n_paths: int


def atr(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def evaluate(symbol, hist, paths, min_move_atr, min_agreement, min_rr, horizon_bars,
             sl_min_atr=0.5, tp_rr=None):
    """Kronos-only signal evaluation: forecast path agreement and forecast-derived risk/reward."""
    entry = float(hist["close"].iloc[-1])
    finals = np.array([float(p["close"].iloc[-1]) for p in paths], dtype=float)
    if len(finals) < 2 or not np.isfinite(finals).all():
        return None, f"{symbol}: insufficient/invalid forecast paths"
    rets = finals / entry - 1.0
    mean_move = float(rets.mean()) * 100
    p_up = float((rets > 0).mean())
    long = mean_move > 0
    agreement = p_up if long else 1 - p_up
    direction = "LONG" if long else "SHORT"

    atr_val = atr(hist)
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None, f"{symbol}: invalid ATR"
    required = min_move_atr * atr_val / entry * 100
    summary = (f"{symbol}: exp {mean_move:+.2f}% (need {required:.2f}%) | "
               f"{direction} Kronos path agreement {agreement:.0%} ({len(paths)} paths)")
    if abs(mean_move) < required:
        return None, summary + " | move too small"
    if agreement < min_agreement:
        return None, summary + f" | agreement < {min_agreement:.0%}"

    highs = np.array([float(p["high"].max()) for p in paths], dtype=float)
    lows = np.array([float(p["low"].min()) for p in paths], dtype=float)
    floor = sl_min_atr * atr_val
    if long:
        sl = min(float(np.percentile(lows, 20)), entry - floor)
        risk = entry - sl
        tp = entry + tp_rr * risk if tp_rr else float(np.median(highs))
        reward = tp - entry
    else:
        sl = max(float(np.percentile(highs, 80)), entry + floor)
        risk = sl - entry
        tp = entry - tp_rr * risk if tp_rr else float(np.median(lows))
        reward = entry - tp
    if reward <= 0 or risk <= 0:
        return None, summary + " | no valid TP/SL"
    rr = float(tp_rr) if tp_rr else reward / risk
    if rr < min_rr - 1e-9:
        return None, summary + f" | R:R {rr:.2f} < {min_rr}"

    return Signal(symbol, direction, entry, tp, sl, rr, mean_move, agreement,
                  horizon_bars, len(paths)), summary + f" | R:R {rr:.2f} -> SIGNAL"
