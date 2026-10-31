from __future__ import annotations
import time
import pandas as pd
import requests

BASE = "https://api.india.delta.exchange"
RES_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


def fetch_candles(symbol: str, resolution: str, start_ts: int, end_ts: int, getter=None) -> pd.DataFrame:
    get = getter or requests.get
    step = RES_MIN[resolution] * 60
    chunk = 1500 * step
    rows: list[dict] = []
    s = int(start_ts)
    while s < end_ts:
        e = min(s + chunk, int(end_ts))
        data = None
        for attempt in range(4):
            try:
                r = get(f"{BASE}/v2/history/candles",
                        params={"resolution": resolution, "symbol": symbol, "start": s, "end": e},
                        timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"Delta API error for {symbol}: {data}")
        rows += (data or {}).get("result") or []
        if e >= end_ts:
            break
        s = e - step
        time.sleep(0.25)
    if not rows:
        raise ValueError(f"no candles returned for {symbol}")
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="s")
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    df.index.name = None
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def regularize(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    full = pd.date_range(df.index[0], df.index[-1], freq=f"{minutes}min")
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df["amount"] = df["close"] * df["volume"]
    return df
