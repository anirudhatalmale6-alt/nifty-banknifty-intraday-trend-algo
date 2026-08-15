# Nifty / Bank Nifty — Intraday Trend-Following Algorithm

A complete, runnable intraday trend-following system for the Nifty 50 and Nifty
Bank indices: strategy code, an event-driven backtester with a realistic Indian
cost model, a self-contained HTML performance report, a matching TradingView
Pine Script, and a test suite that proves the engine does what this document
says it does.

**Everything here runs on your machine with two commands and no API keys.**

---

## Contents

| Path | What it is |
|---|---|
| `run_backtest.py` | Command-line entry point — the thing you actually run |
| `config/nifty.yaml` | Every tunable parameter for Nifty 50 |
| `config/banknifty.yaml` | Same, tuned for Bank Nifty's wider range |
| `src/nifty_algo/indicators.py` | EMA, ATR, ADX, Supertrend, Donchian — all causal |
| `src/nifty_algo/strategy.py` | The trading rules, and nothing else |
| `src/nifty_algo/backtest.py` | Event-driven simulator: fills, stops, costs, risk limits |
| `src/nifty_algo/metrics.py` | Every performance statistic in the report |
| `src/nifty_algo/report.py` | The HTML report generator |
| `src/nifty_algo/data.py` | Yahoo Finance / CSV loading, session filtering, caching |
| `tools/sensitivity.py` | Parameter grid, cost sensitivity, out-of-sample split |
| `pine/nifty_trend_strategy.pine` | TradingView version of the same logic |
| `tests/` | 82 tests, including the no-lookahead tripwires |
| `output/` | Generated reports land here |

---

## 1. Setup

Python 3.10 or newer. From the project folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That is the whole installation. Five packages, all standard:
`pandas`, `numpy`, `yfinance`, `PyYAML`, `matplotlib`.

## 2. Run a backtest

```bash
python run_backtest.py --config config/nifty.yaml
python run_backtest.py --config config/banknifty.yaml
```

Each run prints a summary to the terminal and writes four files into `output/`:

| File | Contents |
|---|---|
| `<tag>_report.html` | The full report. Open it in any browser — charts are embedded, nothing loads from the internet |
| `<tag>_trades.csv` | Every trade: entry, exit, size, stop, gross, costs, net, R multiple, exit reason |
| `<tag>_equity.csv` | Bar-by-bar equity curve |
| `<tag>_metrics.json` | The metric block, machine readable |

Handy overrides, so you do not have to keep editing the YAML:

```bash
python run_backtest.py --config config/nifty.yaml --interval 5m
python run_backtest.py --config config/nifty.yaml --interval 1h --period 730d
python run_backtest.py --config config/nifty.yaml --capital 2000000 --risk 0.005
python run_backtest.py --config config/nifty.yaml --csv data/my_data.csv --tag mine
python run_backtest.py --config config/nifty.yaml --no-cache      # force fresh download
```

## 3. Run the tests

```bash
pip install pytest
python -m pytest tests/ -q
```

All 82 should pass. They are not decoration — `test_indicator_is_causal` and
`test_results_are_unchanged_by_data_that_arrives_later` are the two that would
catch a lookahead bug, which is the failure mode that makes a backtest look
brilliant and a live account look terrible.

---

## 4. The strategy

Pure trend-following. There is no mean reversion anywhere in the code. It never
buys weakness or sells strength — it joins a move that is already underway and
already confirmed on a slower timeframe.

### Entry — five conditions, all must agree on a bar's close

| # | Condition | Why it is there |
|---|---|---|
| 1 | **Higher timeframe trend** — on the signal timeframe × `htf_multiplier`, fast EMA above slow EMA *and* the slow EMA rising | Which way is the river flowing. Trading against this is the most expensive mistake available |
| 2 | **Supertrend direction** — already flipped to that side on the signal timeframe | The trend's own statement about itself. Doubles as the exit later |
| 3 | **Structure breakout** — close beyond the highest high / lowest low of the previous `breakout_length` bars | The trigger. Without it the system enters mid-range in a market that merely looks trendy |
| 4 | **Trend strength** — ADX at or above `adx_threshold` | ADX ignores direction and measures conviction. Below the threshold, breakouts fail |
| 5 | **Live volatility** — ATR/price at or above `min_atr_pct` | In a dead tape the stop distance collapses, position size explodes, and one tick of noise stops you out |

Plus the session gates: nothing before `entry_start`, nothing after
`entry_cutoff`, everything flat by `square_off`.

Short entries are the exact mirror. Nothing is asymmetric.

### Exit — whichever comes first

- **Initial stop** at `stop_atr_multiplier` × ATR from the entry fill
- **Break-even move** once price runs `breakeven_atr_multiplier` × ATR in your favour
- **Supertrend flip** — a close through the Supertrend line closes the trade
- **EMA cross** back against the position (optional, `exit_on_ema_cross`)
- **Fixed target** at `target_atr_multiplier` × ATR — *off by default*, because a
  trend system earns its living on the tail and capping the tail is expensive
- **Hard square-off** at `square_off`. Nothing is ever carried overnight

### Risk management

| Control | Default | What it does |
|---|---|---|
| `risk_per_trade_pct` | 1% | Position size is derived from this and the stop distance, not fixed |
| `max_risk_per_trade_pct` | 3% | Hard ceiling. A trade that would exceed it is skipped, full stop |
| `max_leverage` | 5× | Notional exposure ceiling |
| `daily_loss_limit_pct` | 2% | Once the day's realised P&L hits this, no new entries. The session is over |
| `max_trades_per_day` | 3 | Overtrading is how an intraday edge gets paid to the exchange |
| `max_open_positions` | 1 | One position at a time. This is not a portfolio strategy |
| `allow_min_one_lot` | true | Derivatives come in indivisible lots. If the risk-based size rounds to zero, take one lot — but only if that stays inside `max_risk_per_trade_pct` |

### Two switches worth understanding

**`entry_mode`** — `donchian` (default) breaks out of the last N bars and carries
structure across sessions. `opening_range` breaks out of the first
`opening_range_minutes` of the day; entries happen earlier, so the trade has
more of the session left before square-off. Both are gated by exactly the same
five filters — only the trigger differs.

**`supertrend_trail_mode`** — `close` (default) exits when a bar *closes* through
the Supertrend line. `intrabar` turns the line into a resting stop order that
fires the moment price touches it. `intrabar` locks in more but converts a lot
of eventual winners into small losses. This one switch moves the results more
than any other parameter in the file.

---

## 5. How the backtester avoids lying to you

Five rules, enforced in code and covered by tests:

1. A signal is only ever read from a bar that has **closed**.
2. Entries and signal-based exits fill at the **next bar's open**, with slippage.
3. Stop and target hits are checked against the **current bar's** high/low and
   fill at the stop price — never at a better one.
4. When one bar's range contains both the stop and the target, the order they
   were touched in is unknowable. With `pessimistic_intrabar: true` the engine
   **always assumes the stop**.
5. The trailing stop for bar *i+1* is computed from bar *i*'s close. It is never
   tightened using information from inside bar *i+1*.

Costs are modelled the way an Indian broker actually charges: flat brokerage per
order, exchange transaction charges, STT on the sell side, SEBI turnover fee,
stamp duty on the buy side, GST on the brokerage component — plus a separate
slippage assumption applied to the fill price. All of it is in `config/*.yaml`
under `costs:`, and all of it is visible in the report.

---

## 6. Using your own data

Yahoo Finance is free and needs no setup, but its intraday history is capped:

| Interval | How far back Yahoo will go |
|---|---|
| 1m | 7 days |
| 5m, 15m, 30m | **60 days** |
| 1h | 730 days |
| 1d | Decades |

Sixty days of 15-minute bars is roughly 58 trading sessions and 30 trades. That
is enough to prove the code works. **It is not enough to prove the strategy
works** — see the findings below.

For a multi-year intraday backtest, export data from your broker (Zerodha Kite,
Upstox, Fyers, Dhan and TrueData all provide historical intraday APIs) and point
the config at the CSV:

```yaml
data:
  csv_path: "data/nifty_5min_2019_2026.csv"
```

Required format — column names are case-insensitive, extra columns are ignored:

```csv
datetime,open,high,low,close,volume
2024-01-01 09:15:00,21730.75,21755.60,21725.30,21741.90,0
2024-01-01 09:20:00,21742.10,21768.45,21738.00,21762.35,0
```

Timestamps without a timezone are assumed to be IST, which is what every Indian
broker exports. Nothing else in the system needs to change.

### Reproducibility

Downloads are cached to `data/cache/` as CSV, and **the cache files used to
generate every report in `output/` are committed to this repository.** That is
deliberate: Yahoo's 60-day intraday window slides forward every single day, so
without the cache your run next week would silently use different bars and
produce different numbers.

Because the cache ships with the code, `./run_all.sh` on your machine reproduces
the committed reports **exactly** — same trades, same P&L, to the rupee. Verify
it by diffing `output/nifty_15m_trades.csv` against the copy in git after a run;
it should come back clean.

To pull fresh data instead, pass `--no-cache` or delete the relevant file from
`data/cache/`. The exact configuration behind any report is printed at the
bottom of that report's HTML.

---

## 7. Findings from the runs in `output/`

Reported straight, including the parts that are not flattering.

**The logic works. The friction is the problem.**

On 60 days of 15-minute Nifty data the strategy generated a gross profit of
about ₹58,000 on ₹10 lakh of capital across 32 trades — a profit factor of 1.55
before costs. After realistic brokerage, statutory charges and 2 basis points of
slippage, the net was ₹1,652. Friction took essentially the entire edge.

Run it yourself:

```bash
python tools/sensitivity.py costs --config config/nifty.yaml
```

| Scenario | Gross | Charges | Net | PF |
|---|---:|---:|---:|---:|
| Frictionless (not achievable) | 57,892 | 0 | 57,892 | 1.55 |
| Statutory charges, zero slippage | 63,153 | 30,079 | 33,074 | 1.27 |
| **As configured (2 bps slippage)** | **31,729** | **30,076** | **1,652** | **1.01** |
| Double slippage (4 bps) | 998 | 29,597 | −28,599 | 0.82 |
| Triple slippage (6 bps) | −38,164 | 29,593 | −67,757 | 0.65 |

Two things follow from this table, and they are the most useful output of the
whole project:

- **Your real slippage decides whether this is profitable.** Not the parameters.
  Measure your actual fills against the mid before you trust any of these
  numbers, then put the measured figure in `slippage_pct`.
- **Turnover is the enemy.** Every extra trade pays roughly ₹1,000 of friction on
  a ₹36 lakh notional. Filters that reduce trade count are worth more than
  filters that improve win rate.

**Shorter timeframes are worse, not better.** On 5-minute bars the same logic
lost 19–23% over the same 60 days on both indices. Roughly 2.5× the trades, 2.5×
the friction, and no increase in edge per trade. If you want to trade 5-minute
bars, the entry filters need to be far stricter than the defaults.

**Hourly bars do not suit an intraday mandate.** Over 730 days of 1-hour data,
75–80% of trades exited on the forced square-off rather than on a stop or a
trend reversal — the trades simply never had room to develop inside a 6¼-hour
session. 15 minutes is the sweet spot for this design.

**The core logic does have edge on a longer sample.** Run on 15 years of daily
bars — where the square-off constraint disappears — the same rules produced a
profit factor of 1.10 on Nifty and 1.92 on Bank Nifty across 82 and 59 trades.
That is what tells you the entry logic is sound and the intraday result is a
friction-and-runway problem, not a broken signal.

**The 60-day sample is too short to tune on, and here is the proof.**

```bash
python tools/sensitivity.py oos --config config/nifty.yaml
```

Fitting the parameter grid on the first 70% of sessions and testing on the last
30%:

| | Train | Test |
|---|---:|---:|
| Return | +4.58% | −1.23% |
| Sharpe | 1.84 | −1.13 |
| Profit factor | 1.62 | 0.70 |

The best in-sample parameters fall apart out of sample. **This is the expected
result on 58 sessions, and it is why the shipped defaults are not the grid
winners.** They were chosen on principle — standard indicator periods, a stop
wide enough to survive normal noise, filters strict enough to keep turnover
down — rather than fitted to this particular window. Fitting to it would
manufacture a beautiful backtest and a worthless strategy.

Only 33 of 192 grid combinations were profitable on this sample. A healthy
strategy shows a broad connected region of profitable parameters; this shows
scattered cells. On 60 days of data that is uninformative rather than damning —
but it is not evidence of an edge, and it should not be presented as one.

### What would change the picture

1. **Real intraday history.** Two to five years of 5-minute or 15-minute bars
   from your broker. Everything above is a 58-session sample; that is a coin
   flip, not a conclusion. The loader already accepts your CSV.
2. **Your measured slippage.** The single most load-bearing number in the config.
3. **Your actual instrument.** These runs use index values. You will trade
   futures (with a spread and a roll) or options (with a spread and theta). The
   cost block needs to reflect whichever you use.

---

## 8. TradingView version

`pine/nifty_trend_strategy.pine` is a bar-for-bar mirror of the Python logic in
Pine Script v5.

1. TradingView → Pine Editor → paste the file → Save → **Add to chart**
2. Set the chart to **15 minutes**, symbol `NSE:NIFTY` or `NSE:BANKNIFTY`
3. Match the inputs to your YAML, then open **Strategy Tester**

The signals are identical. The P&L is not, for four reasons documented at the
top of the Pine file: Pine uses a fixed quantity rather than risk-based sizing,
anchors the stop to the signal bar's close rather than the fill, uses a flat
commission rather than the full Indian charge stack, and runs on TradingView's
data feed rather than Yahoo's.

**Use Pine to see that the logic is right. Use the Python report for the number
you act on.**

Note the `lookahead_off` flag on every `request.security` call. Without it, Pine
hands you a finished higher-timeframe bar while it is still forming — the single
most common way a Pine strategy fakes a great backtest.

---

## 9. Going live

This project backtests; it does not place orders. That is deliberate — you
should forward-test on paper before any code of mine touches your broker
account. When you are ready, the sequence is:

1. **Paper-trade the Pine alerts for a month.** `alertcondition` hooks are
   already in the script. Compare the alerts against the Python signals on the
   same bars.
2. **Log your real slippage.** Every fill, against the mid at signal time. Put
   the measured average into `slippage_pct` and re-run the backtest. This is the
   step most people skip and most people regret.
3. **Then wire up execution.** The signal layer (`strategy.prepare`) is already
   separate from the simulation layer, so a live loop is: pull the latest bars,
   call `prepare()`, read the last row's `signal`, place the order. Broker
   integration (Kite Connect, Fyers, Dhan) is a separate piece of work and needs
   its own error handling, reconciliation and kill switch.

---

## Disclaimer

This is software, not financial advice. A backtest is a simulation and not a
promise. Results depend entirely on the data vendor, the assumed slippage and
the cost model — change any one of them and the numbers change, as section 7
demonstrates in detail. Index values are not directly tradeable; real fills
happen in futures or options, which carry their own spread, roll and decay.
Trading leveraged intraday derivatives can lose you more than your capital.
Forward-test on paper first.
