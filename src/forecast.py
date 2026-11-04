from __future__ import annotations
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "Kronos"))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

COLS = ["open", "high", "low", "close", "volume", "amount"]


class Forecaster:
    def __init__(self, model_name: str, tokenizer_name: str, max_context: int = 512, device: str | None = None):
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        kwargs = {"max_context": max_context}
        if device:
            kwargs["device"] = device
        self.predictor = KronosPredictor(model, tokenizer, **kwargs)

    @staticmethod
    def _inputs(df: pd.DataFrame, interval_min: int, pred_len: int):
        x_df = df[COLS].reset_index(drop=True)
        x_ts = pd.Series(df.index)
        start = df.index[-1] + pd.Timedelta(minutes=interval_min)
        y_ts = pd.Series(pd.date_range(start, periods=pred_len, freq=f"{interval_min}min"))
        return x_df, x_ts, y_ts

    def sample_paths_batch(self, frames: list[pd.DataFrame], interval_min: int, pred_len: int,
                           n_paths: int, T: float = 1.0, top_p: float = 0.9) -> list[list[pd.DataFrame]]:
        """Forecast all symbols in one predict_batch call; returns n_paths per input frame."""
        if not frames:
            return []
        inputs = [self._inputs(df, interval_min, pred_len) for df in frames]
        df_list, x_timestamps, y_timestamps = [], [], []
        for x_df, x_ts, y_ts in inputs:
            for _ in range(n_paths):
                df_list.append(x_df)
                x_timestamps.append(x_ts)
                y_timestamps.append(y_ts)

        predictions = self.predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_timestamps,
            y_timestamp_list=y_timestamps,
            pred_len=pred_len, T=T, top_p=top_p, sample_count=1, verbose=False,
        )
        expected = len(frames) * n_paths
        if len(predictions) != expected:
            raise RuntimeError(f"Kronos returned {len(predictions)} paths; expected {expected}")
        return [predictions[i * n_paths:(i + 1) * n_paths] for i in range(len(frames))]

    def sample_paths(self, df: pd.DataFrame, interval_min: int, pred_len: int,
                     n_paths: int, T: float = 1.0, top_p: float = 0.9) -> list[pd.DataFrame]:
        """Backward-compatible single-symbol wrapper around the batch API."""
        return self.sample_paths_batch([df], interval_min, pred_len, n_paths, T, top_p)[0]
