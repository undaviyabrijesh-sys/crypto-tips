from __future__ import annotations
import time

import pandas as pd
import yfinance as yf

from .delta import RES_MIN, fetch_candles as delta_candles, regularize


def interval_minutes(interval: str) -> int:
    return RES_MIN[interval]


def _from_delta(symbols: list[str], interval: str, lookback: int) -> tuple[pd.DataFrame, str]:
    mins = interval_minutes(interval)
    step = mins * 60
    end_ts = int(time.time() // step) * step              # start of the still-forming candle
    start_ts = end_ts - (lookback + 30) * step
    last_err: Exception | None = None
    for s in symbols:
        try:
            df = delta_candles(s, interval, start_ts, end_ts)
            df = df[df.index < pd.Timestamp(end_ts, unit="s")]
            df = regularize(df, mins)
            if len(df) < lookback:
                raise ValueError(f"{s}: only {len(df)} candles, need {lookback}")
            return df.iloc[-lookback:], f"Delta {s}"
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Delta failed: {last_err}")


def _from_yahoo(tickers: list[str], interval: str, lookback: int) -> tuple[pd.DataFrame, str]:
    mins = interval_minutes(interval)
    last_err: Exception | None = None
    for t in tickers:
        try:
            raw = yf.Ticker(t).history(period="30d", interval=interval, auto_adjust=False)
            if raw is None or raw.empty:
                raise ValueError(f"no data for {t}")
            df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
            idx = df.index
            idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
            df.index = idx.tz_localize(None)
            df = df.dropna(subset=["open", "high", "low", "close"])
            df["volume"] = df["volume"].fillna(0.0)
            df["amount"] = df["close"] * df["volume"]
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            if len(df) and df.index[-1] + pd.Timedelta(minutes=mins) > now:
                df = df.iloc[:-1]
            if len(df) < lookback:
                raise ValueError(f"{t}: only {len(df)} candles, need {lookback}")
            return df.iloc[-lookback:], f"Yahoo {t}"
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Yahoo failed: {last_err}")


def fetch_candles(tickers: list[str], interval: str, lookback: int,
                  delta_symbols: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Completed candles, UTC-naive index. Delta Exchange India first, Yahoo as fallback."""
    if delta_symbols:
        try:
            return _from_delta(delta_symbols, interval, lookback)
        except Exception as e:
            print(f"  (Delta unavailable, using Yahoo: {e})")
    return _from_yahoo(tickers, interval, lookback)


def is_stale(df: pd.DataFrame, interval: str, max_gap_bars: int = 3) -> bool:
    mins = interval_minutes(interval)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    close_time = df.index[-1] + pd.Timedelta(minutes=mins)
    return (now - close_time) > pd.Timedelta(minutes=mins * max_gap_bars)
