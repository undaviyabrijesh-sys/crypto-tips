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


def evaluate(symbol, hist, paths, min_move_atr, min_agreement, min_rr, horizon_bars):
    entry = float(hist["close"].iloc[-1])
    finals = np.array([float(p["close"].iloc[-1]) for p in paths])
    rets = finals / entry - 1.0
    mean_move = float(rets.mean()) * 100
    p_up = float((rets > 0).mean())
    long = mean_move > 0
    agreement = p_up if long else 1 - p_up
    direction = "LONG" if long else "SHORT"

    atr_val = atr(hist)
    required = min_move_atr * atr_val / entry * 100
    summary = (f"{symbol}: exp {mean_move:+.2f}% (need {required:.2f}%) | "
               f"{direction} agreement {agreement:.0%}")
    if abs(mean_move) < required:
        return None, summary + " | move too small"
    if agreement < min_agreement:
        return None, summary + f" | agreement < {min_agreement:.0%}"

    highs = np.array([float(p["high"].max()) for p in paths])
    lows = np.array([float(p["low"].min()) for p in paths])
    floor = 0.5 * atr_val
    if long:
        tp = float(np.median(highs))
        sl = min(float(np.percentile(lows, 20)), entry - floor)
        reward, risk = tp - entry, entry - sl
    else:
        tp = float(np.median(lows))
        sl = max(float(np.percentile(highs, 80)), entry + floor)
        reward, risk = entry - tp, sl - entry
    if reward <= 0 or risk <= 0:
        return None, summary + " | no valid TP/SL"
    rr = reward / risk
    if rr < min_rr:
        return None, summary + f" | R:R {rr:.2f} < {min_rr}"

    return Signal(symbol, direction, entry, tp, sl, rr, mean_move, agreement,
                  horizon_bars, len(paths)), summary + f" | R:R {rr:.2f} -> SIGNAL"
