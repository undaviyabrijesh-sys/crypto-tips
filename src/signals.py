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


def _ema_bias(df: pd.DataFrame) -> int:
    """Return +1 bullish, -1 bearish, 0 neutral/insufficient history."""
    if df is None or len(df) < 55 or "close" not in df:
        return 0
    close = df["close"].astype(float)
    e20, e50 = close.ewm(span=20, adjust=False).mean().iloc[-1], close.ewm(span=50, adjust=False).mean().iloc[-1]
    px = float(close.iloc[-1])
    if px > e20 > e50:
        return 1
    if px < e20 < e50:
        return -1
    return 0


def trend_alignment(direction: str, frames: dict | None) -> tuple[int, str]:
    """Score direction agreement across 15m, 1h and 4h (0..3).
    The 4h frame is a regime guard: an opposite 4h trend blocks the setup.
    """
    if not frames:
        return 0, "trend data unavailable"
    wanted = 1 if direction == "LONG" else -1
    biases = [_ema_bias(frames.get(k)) for k in ("15m", "1h", "4h")]
    score = sum(b == wanted for b in biases)
    labels = [("bull" if b == 1 else "bear" if b == -1 else "neutral") for b in biases]
    return score, f"trend 15m/1h/4h={'/'.join(labels)}"


def evaluate(symbol, hist, paths, min_move_atr, min_agreement, min_rr, horizon_bars,
             sl_min_atr=0.5, tp_rr=None, trend_frames=None, min_trend_score=2):
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
    score, trend_note = trend_alignment(direction, trend_frames)
    summary = (f"{symbol}: exp {mean_move:+.2f}% (need {required:.2f}%) | "
               f"{direction} path agreement {agreement:.0%} ({len(paths)} paths) | "
               f"{trend_note} [{score}/3]")
    if abs(mean_move) < required:
        return None, summary + " | move too small"
    if agreement < min_agreement:
        return None, summary + f" | path agreement < {min_agreement:.0%}"
    if trend_frames is not None:
        biases_4h = _ema_bias(trend_frames.get("4h"))
        wanted = 1 if long else -1
        if biases_4h == -wanted:
            return None, summary + " | blocked: 4h trend opposes forecast"
        if score < min_trend_score:
            return None, summary + f" | trend score < {min_trend_score}/3"

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
