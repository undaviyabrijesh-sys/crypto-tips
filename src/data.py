from __future__ import annotations
import pandas as pd
import yfinance as yf

INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60}


def interval_minutes(interval: str) -> int:
    return INTERVAL_MIN[interval]


def fetch_candles(tickers: list[str], interval: str, lookback: int) -> tuple[pd.DataFrame, str]:
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
            return df.iloc[-lookback:], t
        except Exception as e:
            last_err = e
    raise RuntimeError(f"all tickers failed: {last_err}")


def is_stale(df: pd.DataFrame, interval: str, max_gap_bars: int = 3) -> bool:
    mins = interval_minutes(interval)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    close_time = df.index[-1] + pd.Timedelta(minutes=mins)
    return (now - close_time) > pd.Timedelta(minutes=mins * max_gap_bars)
