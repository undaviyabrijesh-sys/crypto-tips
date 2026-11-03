# Kronos / crypto-tips research plan

## Scope

This repository remains independent from DMPE. The first target is to understand why
Kronos alerts sometimes choose the wrong direction and why trades hit their stops.
No live order placement is added. Discord alerts remain informational.

## Baseline to preserve

- 15-minute candles; 400-candle context; 16-candle (4-hour) forecast horizon.
- Kronos-base with the current tokenizer.
- Current live thresholds and exit rules are recorded as the baseline; do not silently
  tune them while evaluating a candidate.
- Current default sample count is eight paths. Treat its agreement score as a coarse
  fraction (increments of 12.5 percentage points), not a calibrated probability.

## Research sequence

1. **Reproduce the baseline.** Run a BTC-only historical test first, then extend the
   date range. Save raw forecast paths/cache and the exact configuration with results.
2. **Separate prediction from execution.** Measure directional hit rate over the
   forecast horizon without stops. Then simulate entries and exits using the same
   timestamps. This tells us whether the failure is forecast direction or trade
   construction.
3. **Diagnose stopped trades.** Record adverse/favourable excursion before the exit
   candle and whether price reaches the original target later in the horizon after a
   stop. A high stopped-then-target rate is evidence that the stop placement deserves
   investigation; it is not by itself proof that widening the stop is profitable.
4. **Test direction filters.** Compare the unfiltered forecast with predeclared trend
   and volatility filters. Validate thresholds on an earlier chronological segment;
   keep a later segment untouched for final evaluation.
5. **Test risk placement separately.** Compare the current forecast-extreme stop with
   ATR and recent-swing stops. Compare fixed 2R and 3R targets. Keep position risk
   constant when stop distance changes.
6. **Only promote changes that replicate.** Report trade count, coverage, directional
   accuracy, precision by confidence bucket, win rate, profit factor, net expectancy
   after fees/slippage, maximum drawdown, and regime breakdown. Do not select the
   single best grid cell from one short sample.

## Important interpretation limits

- Eight sampled paths are not enough to treat path agreement as a reliable probability.
  Increase path count only in a controlled experiment and compare repeatability.
- OHLC bars do not reveal intrabar barrier order. Ambiguous stop/target touches are
  treated conservatively as stop-first.
- Stop gaps are filled at the candle open, not the stop price.
- The after_stop_target statistic is a diagnostic of post-stop price movement within
  the original horizon. It is not a tradable exit rule.
- Backtests are research estimates, not evidence of future profitability. Include
  exchange fees, GST treatment, slippage, funding where applicable, and data-source
  differences before paper trading.
