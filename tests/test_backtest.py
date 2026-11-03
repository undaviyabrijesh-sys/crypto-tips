import unittest

import pandas as pd

from backtest import direction_stats, metrics, simulate


class ExecutionSimulationTests(unittest.TestCase):
    def candle(self, open_, high, low, close):
        return pd.DataFrame([{"open": open_, "high": high, "low": low, "close": close}])

    def test_long_stop_gap_fills_at_open_not_stop(self):
        px, reason, bar = simulate("LONG", 100, 110, 95, self.candle(92, 96, 90, 93))
        self.assertEqual((px, reason, bar), (92.0, "SL", 0))

    def test_short_stop_gap_fills_at_open_not_stop(self):
        px, reason, bar = simulate("SHORT", 100, 90, 105, self.candle(108, 110, 104, 106))
        self.assertEqual((px, reason, bar), (108.0, "SL", 0))

    def test_long_target_gap_gets_price_improvement(self):
        px, reason, bar = simulate("LONG", 100, 110, 95, self.candle(112, 115, 109, 114))
        self.assertEqual((px, reason, bar), (112.0, "TP", 0))

    def test_short_target_gap_gets_price_improvement(self):
        px, reason, bar = simulate("SHORT", 100, 90, 105, self.candle(88, 91, 85, 86))
        self.assertEqual((px, reason, bar), (88.0, "TP", 0))

    def test_ambiguous_intrabar_barrier_touch_is_stop_first(self):
        px, reason, bar = simulate("LONG", 100, 110, 95, self.candle(100, 111, 94, 105))
        self.assertEqual((px, reason, bar), (95, "SL", 0))


class BacktestMetricsTests(unittest.TestCase):
    def test_drawdown_includes_initial_zero_equity(self):
        trades = pd.DataFrame([{
            "net_R": -1.0, "mirror_net_R": 1.0, "risk_pct": 1.0, "reason": "SL",
            "mfe_prior_exit_R": 0.0, "mae_prior_exit_R": 0.0, "after_stop_target": True,
        }, {
            "net_R": 2.0, "mirror_net_R": -2.0, "risk_pct": 1.0, "reason": "TP",
            "mfe_prior_exit_R": 1.0, "mae_prior_exit_R": 0.0, "after_stop_target": False,
        }])
        result = metrics(trades)
        self.assertEqual(result["max_dd_R"], 1.0)
        self.assertEqual(result["stopped_then_target_%"], 100.0)

    def test_empty_metrics_are_well_formed(self):
        result = metrics(pd.DataFrame())
        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["stopped_then_target_%"], 0.0)

    def test_direction_confidence_matches_live_mean_direction(self):
        # Five modest up paths are outweighed by three large down paths.
        # The mean forecast is short, but only 3/8 paths agree with it.
        closes = [100.0, 101.0, 100.0, 99.0]
        data = {"BTCUSD": pd.DataFrame({"close": closes})}
        steps = {"BTCUSD": [{
            "i": 2,
            "finals": [102.0] * 5 + [80.0] * 3,
        }]}
        cfg = {"forecast": {"horizon_bars": 1}}
        rows, _, _ = direction_stats(data, steps, cfg, min_agreement=0.75)
        self.assertEqual(rows[0]["agreement"], "under 62%")
        self.assertEqual(rows[0]["windows"], 1)


if __name__ == "__main__":
    unittest.main()
