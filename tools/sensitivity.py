#!/usr/bin/env python3
"""
Parameter and cost sensitivity study.

A single backtest tells you what one set of parameters did on one slice of
history. That is the least useful thing a backtester can tell you. What you
actually want to know is:

  * Is the result a broad plateau, or a single lucky spike? A strategy that only
    works at ADX 27.5 does not work.
  * How much of the gross edge does friction eat? On Indian intraday indices
    this is usually the whole story.
  * Does it survive out-of-sample? Fit on the first 70%, test on the last 30%.

Usage
    python tools/sensitivity.py grid   --config config/nifty.yaml
    python tools/sensitivity.py costs  --config config/nifty.yaml
    python tools/sensitivity.py oos    --config config/nifty.yaml

Add --interval / --period to point any of them at different data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nifty_algo import (  # noqa: E402
    Config, filter_session, is_intraday_interval, load_ohlcv, prepare, run_backtest,
)
from nifty_algo import metrics as mx  # noqa: E402

log = logging.getLogger("sensitivity")

# The grid. Deliberately coarse and wide: the point is to see the shape of the
# surface, not to find its exact peak. Finding the exact peak is overfitting.
GRID = {
    "adx_threshold": [15.0, 20.0, 25.0, 30.0],
    "breakout_length": [8, 12, 20, 30],
    "stop_atr_multiplier": [1.0, 1.5, 2.0, 2.5],
    "supertrend_multiplier": [2.0, 2.5, 3.0],
}


def load_bars(cfg: Config) -> pd.DataFrame:
    raw = load_ohlcv(cfg.data)
    return filter_session(raw, cfg.session, is_intraday_interval(cfg.data.interval))


def one_run(cfg: Config, bars: pd.DataFrame) -> dict:
    intraday = is_intraday_interval(cfg.data.interval)
    res = run_backtest(prepare(bars, cfg, intraday), cfg, intraday)
    return mx.compute_metrics(res.trades, res.equity_curve, cfg.risk.starting_capital)


def set_strategy(cfg: Config, **kwargs) -> Config:
    for k, v in kwargs.items():
        setattr(cfg.strategy, k, v)
    cfg.strategy.__post_init__()  # re-validate after the mutation
    return cfg


# --------------------------------------------------------------------------
def cmd_grid(cfg: Config, bars: pd.DataFrame, args) -> None:
    keys = list(GRID)
    rows = []
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    log.info("Running %d parameter combinations ...", len(combos))

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        m = one_run(set_strategy(cfg, **params), bars)
        rows.append({**params,
                     "return_pct": m["total_return_pct"],
                     "max_dd_pct": m["max_drawdown_pct"],
                     "sharpe": m["sharpe"],
                     "trades": m["total_trades"],
                     "win_rate": m["win_rate_pct"],
                     "profit_factor": m["profit_factor"]})
        if i % 20 == 0:
            log.info("  %d/%d", i, len(combos))

    df = pd.DataFrame(rows).sort_values("return_pct", ascending=False)
    out = Path(cfg.backtest.output_dir) / f"{args.tag}_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    positive = (df["return_pct"] > 0).sum()
    print(f"\nTop 15 of {len(df)} combinations\n" + "-" * 78)
    print(df.head(15).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("-" * 78)
    print(f"{positive}/{len(df)} combinations profitable "
          f"({positive / len(df) * 100:.0f}%)")
    print("\nRead this as a robustness check, not a shopping list. If only a "
          "handful\nof cells are green, the strategy is not working on this "
          "data — the green\ncells are noise. You want a broad connected "
          "region, and you want to sit\nin the middle of it, not on its best "
          "cell.")
    print(f"\nFull grid: {out}")


def cmd_costs(cfg: Config, bars: pd.DataFrame, args) -> None:
    """How much of the gross edge does friction take?"""
    scenarios = [
        ("Frictionless (upper bound, not achievable)", dict(
            slippage_pct=0.0, brokerage_per_order=0.0, stt_sell_pct=0.0,
            exchange_txn_pct=0.0, stamp_duty_pct=0.0, sebi_fee_pct=0.0, gst_pct=0.0)),
        ("Statutory charges only, zero slippage", dict(slippage_pct=0.0)),
        ("Half the configured slippage", dict(slippage_pct=cfg.costs.slippage_pct / 2)),
        ("As configured", dict()),
        ("Double the configured slippage", dict(slippage_pct=cfg.costs.slippage_pct * 2)),
        ("Triple the configured slippage", dict(slippage_pct=cfg.costs.slippage_pct * 3)),
    ]

    base = {k: getattr(cfg.costs, k) for k in
            ("slippage_pct", "brokerage_per_order", "stt_sell_pct",
             "exchange_txn_pct", "stamp_duty_pct", "sebi_fee_pct", "gst_pct")}

    rows = []
    for label, overrides in scenarios:
        for k, v in {**base, **overrides}.items():
            setattr(cfg.costs, k, v)
        m = one_run(cfg, bars)
        rows.append({
            "scenario": label,
            "gross_pnl": m["gross_profit"],
            "charges": m["total_costs"],
            "net_pnl": m["net_profit"],
            "return_pct": m["total_return_pct"],
            "trades": m["total_trades"],
            "profit_factor": m["profit_factor"],
        })

    for k, v in base.items():  # restore
        setattr(cfg.costs, k, v)

    df = pd.DataFrame(rows)
    out = Path(cfg.backtest.output_dir) / f"{args.tag}_costs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\nCost sensitivity\n" + "-" * 98)
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("-" * 98)
    print("\nNote that 'gross_pnl' changes between rows even though the signals "
          "do not.\nThat is slippage moving the fill price, which changes where "
          "stops are hit and\nhow the trade plays out — not just a fee deducted "
          "at the end. Slippage is a\nstrategy input, not an accounting entry.")
    print(f"\nSaved: {out}")


def cmd_oos(cfg: Config, bars: pd.DataFrame, args) -> None:
    """Fit on the first 70% of sessions, test on the last 30%.

    This is the cheap version of a walk-forward. It will not save you from a
    determined overfitter, but it will catch the obvious cases.
    """
    days = sorted(set(bars.index.date))
    if len(days) < 20:
        print("Need at least 20 sessions for a meaningful split.")
        return
    split_day = days[int(len(days) * 0.7)]
    train = bars[bars.index.date < split_day]
    test = bars[bars.index.date >= split_day]
    print(f"Train: {len(train)} bars to {split_day}   |   "
          f"Test: {len(test)} bars from {split_day}")

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    log.info("Fitting %d combinations on the training window ...", len(combos))

    scored = []
    for combo in combos:
        params = dict(zip(keys, combo))
        m = one_run(set_strategy(cfg, **params), train)
        if m["total_trades"] >= 10:  # ignore combinations that barely traded
            scored.append((m["total_return_pct"], params, m))

    if not scored:
        print("No combination produced at least 10 trades in the training "
              "window. The data is too short to fit anything on.")
        return

    scored.sort(reverse=True, key=lambda x: x[0])
    best_ret, best_params, best_m = scored[0]

    test_m = one_run(set_strategy(cfg, **best_params), test)

    print("\nBest parameters on the TRAINING window")
    print(json.dumps(best_params, indent=2))
    print(f"\n{'':<22}{'train':>14}{'test':>14}")
    for label, key, fmt in [
        ("Return", "total_return_pct", "{:.2f}%"),
        ("Max drawdown", "max_drawdown_pct", "{:.2f}%"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Trades", "total_trades", "{:.0f}"),
        ("Win rate", "win_rate_pct", "{:.1f}%"),
        ("Profit factor", "profit_factor", "{:.2f}"),
    ]:
        print(f"{label:<22}{fmt.format(best_m[key]):>14}{fmt.format(test_m[key]):>14}")

    print("\nIf the test column is dramatically worse than the train column, "
          "the\nparameters were fitted to noise. That is the normal outcome on "
          "a short\nsample, and it is the reason this tool exists.")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["grid", "costs", "oos"])
    p.add_argument("--config", required=True)
    p.add_argument("--symbol")
    p.add_argument("--interval")
    p.add_argument("--period")
    p.add_argument("--csv")
    p.add_argument("--tag")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)-7s %(message)s")
    log.setLevel(logging.INFO)

    cfg = Config.from_yaml(args.config)
    if args.symbol:
        cfg.data.symbol = args.symbol
    if args.interval:
        cfg.data.interval = args.interval
    if args.period:
        cfg.data.period = args.period
    if args.csv:
        cfg.data.csv_path = args.csv
    args.tag = args.tag or f"{Path(args.config).stem}_{cfg.data.interval}"

    bars = load_bars(cfg)
    print(f"Data: {cfg.data.name} {cfg.data.interval}  "
          f"{len(bars)} bars  {bars.index[0]} -> {bars.index[-1]}")

    {"grid": cmd_grid, "costs": cmd_costs, "oos": cmd_oos}[args.command](cfg, bars, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
