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

    def sample_paths(self, df: pd.DataFrame, interval_min: int, pred_len: int,
                     n_paths: int, T: float = 1.0, top_p: float = 0.9) -> list[pd.DataFrame]:
        x_df = df[COLS].reset_index(drop=True)
        x_ts = pd.Series(df.index)
        start = df.index[-1] + pd.Timedelta(minutes=interval_min)
        y_ts = pd.Series(pd.date_range(start, periods=pred_len, freq=f"{interval_min}min"))
        return self.predictor.predict_batch(
            df_list=[x_df] * n_paths,
            x_timestamp_list=[x_ts] * n_paths,
            y_timestamp_list=[y_ts] * n_paths,
            pred_len=pred_len, T=T, top_p=top_p, sample_count=1, verbose=False,
        )
