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
   - EMA trend agrees on at least 2 of 3 completed timeframes (15m, 1h, 4h); an opposing 4h trend blocks the setup
   - suggested TP (median path extreme) / SL (20th-percentile adverse extreme, min 0.5 ATR) gives R:R >= 1.5
4. Discord embed with entry / TP / SL / R:R. Same symbol+direction is muted for 4h.

Tune everything in `config.yaml`. The live signal engine intentionally uses only two confirmation factors: (1) Kronos sampled-path agreement and (2) 15m/1h/4h EMA trend alignment. Path agreement is a model-sample statistic, not a calibrated probability of trade success.

## Setup
```bash
# 1. Discord: Server Settings -> Integrations -> Webhooks -> New Webhook -> Copy URL
# 2. Create the repo and push
git init && git add . && git commit -m "init"
gh repo create kronos-alerts --public --source=. --push
gh secret set DISCORD_WEBHOOK_URL        # paste the webhook URL
# 3. Run once: Actions tab -> "Kronos alerts" -> Run workflow
```
Use a **public** repo: scheduled runs every 15 min would exceed the free minutes of a private one.
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

The manual **Kronos backtest** workflow now defaults to a BTC-only 30-day study with 32 sampled paths at 2-hour forecast intervals. It uses the dedicated `backtest_btc.yaml`; the alert configuration is unchanged. The result artifact includes the forecast cache, trade ledger, equity plot and diagnostic summary. Stop gaps are filled at the candle open in the simulator, and the report includes post-stop target recovery diagnostics.

## Notes
- Kronos is a research forecasting model. Its signals here are **unvalidated** - backtest
  and paper-trade before risking money. This is not financial advice.
- Gold has no tick volume on Yahoo (volume = 0); Kronos accepts that. Gold is skipped when the market is closed.
- Crypto prices come from Yahoo, so they can differ slightly from your exchange (e.g. Delta).
