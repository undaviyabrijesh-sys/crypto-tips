# kronos-alerts

Tracks **BTC, ETH and XAUUSD** with the [Kronos](https://github.com/shiyu-coder/Kronos)
**base** model (`NeoQuasar/Kronos-base`, 102M params) and posts a **Discord** message
when the forecast looks like a trade setup.

## How it works
1. Pulls the last 400 completed 15m candles per symbol (yfinance).
2. Kronos-base samples 32 independent futures for the next 16 candles (4h).
3. A setup is flagged only if all of these hold:
   - average forecast move >= `min_move_pct` for that symbol
   - >= 75% of sampled paths agree on direction
   - suggested TP (median path extreme) / SL (20th-percentile adverse extreme, min 0.5 ATR) gives R:R >= 1.5
4. Discord embed with entry / TP / SL / R:R. Same symbol+direction is muted for 4h.

Tune everything in `config.yaml`. The live signal engine relies on Kronos sampled forecast paths and their directional agreement. No EMA or external trend confirmation is applied. Path agreement is a model-sample statistic, not a calibrated probability of trade success.

## Setup
```bash
# 1. Discord: Server Settings -> Integrations -> Webhooks -> New Webhook -> Copy URL
# 2. Create the repo and push
git init && git add . && git commit -m "init"
gh repo create kronos-alerts --public --source=. --push
gh secret set DISCORD_WEBHOOK_URL        # paste the webhook URL
# 3. Run once: Actions tab -> "Kronos alerts" -> Run workflow
```
The scheduled alert workflow checks every 5 minutes, but only forecasts once for each newly completed 15-minute candle. The runtime stores each symbol's last processed candle and any open signal in `state/last_alerts.json` (persisted by the Actions cache).
GitHub cron can run late or skip runs; for tighter timing run it on a VPS (below).

## Run locally
```bash
bash scripts/setup.sh
pip install torch && pip install -r requirements.txt
export DISCORD_WEBHOOK_URL=...
python main.py --test-discord     # check the webhook
python main.py --dry-run          # forecast, print signals only
python main.py                    # real run (add to cron: */15 * * * *)
```

## Research backtest

The manual **Kronos backtest** workflow defaults to a BTC-only six-month direction-only study with 32 sampled paths at 2-hour forecast intervals. Section 1 is evaluated before any exit grid; trade/exit tuning should wait until directional skill clearly beats the naive benchmark. It uses the dedicated `backtest_btc.yaml`; the alert configuration is unchanged. The result artifact includes the forecast cache and direction-only summary. The live alert checks planned-entry drift against a 0.25 ATR limit and blocks new alerts while a prior alert remains open through TP, SL or its 16-candle horizon. Resolved alert outcomes are appended to the paper log in `state/last_alerts.json`; a downloadable 30-day GitHub Actions artifact is published whenever a trade resolves. Stop gaps are filled at the candle open in the simulator, and the report includes post-stop target recovery diagnostics.

## Notes
- Kronos is a research forecasting model. Its signals here are **unvalidated** - backtest
  and paper-trade before risking money. This is not financial advice.
- Gold has no tick volume on Yahoo (volume = 0); Kronos accepts that. Gold is skipped when the market is closed.
- Crypto prices come from Yahoo, so they can differ slightly from your exchange (e.g. Delta).

 
## Signal lifecycle tracking

Every Discord signal receives a unique Signal ID and is stored as OPEN in
`state/last_alerts.json`. On each newly completed 15-minute candle, Kronos checks
the open signal against subsequent candle OHLC data. The first confirmed TP or SL
closes the signal and sends a separate Discord outcome message with entry, exit,
TP, SL and gross/estimated-net P&L. If neither level is reached within the configured
forecast horizon, the signal closes as TIME (horizon expired). If neither event
has occurred yet, its status remains OPEN. When both TP and SL fall inside the same
candle range, the tracker conservatively records SL first. This lifecycle layer
does not alter Kronos forecast or signal selection logic.
