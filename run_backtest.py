#!/usr/bin/env python3
"""
Command-line entry point.

    python run_backtest.py --config config/nifty.yaml
    python run_backtest.py --config config/banknifty.yaml --interval 5m --period 60d
    python run_backtest.py --config config/nifty.yaml --csv mydata.csv --tag mine

Everything the run needs is in the YAML file; the flags below are overrides for
quick experiments so you do not have to keep editing the config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nifty_algo import (  # noqa: E402
    Config,
    filter_session,
    is_intraday_interval,
    load_ohlcv,
    prepare,
    run_backtest,
    write_report,
)
from nifty_algo import metrics as mx  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest the Nifty / Bank Nifty intraday trend-following strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to the YAML config file")
    p.add_argument("--symbol", help="Override data.symbol (e.g. ^NSEI, ^NSEBANK)")
    p.add_argument("--interval", help="Override data.interval (5m, 15m, 1h, 1d)")
    p.add_argument("--period", help="Override data.period (e.g. 60d, 730d)")
    p.add_argument("--start", help="Override data.start (YYYY-MM-DD)")
    p.add_argument("--end", help="Override data.end (YYYY-MM-DD)")
    p.add_argument("--csv", help="Use this CSV instead of downloading")
    p.add_argument("--capital", type=float, help="Override risk.starting_capital")
    p.add_argument("--risk", type=float,
                   help="Override risk.risk_per_trade_pct (0.01 = 1%%)")
    p.add_argument("--tag", help="Output filename prefix (default: derived from config)")
    p.add_argument("--no-cache", action="store_true",
                   help="Force a fresh download instead of using data/cache")
    p.add_argument("--no-report", action="store_true",
                   help="Print the summary but skip writing the HTML report")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.symbol:
        cfg.data.symbol = args.symbol
    if args.interval:
        cfg.data.interval = args.interval
    if args.period:
        cfg.data.period = args.period
        cfg.data.start = None
        cfg.data.end = None
    if args.start:
        cfg.data.start = args.start
    if args.end:
        cfg.data.end = args.end
    if args.csv:
        cfg.data.csv_path = args.csv
    if args.capital:
        cfg.risk.starting_capital = args.capital
    if args.risk:
        cfg.risk.risk_per_trade_pct = args.risk
    if args.no_cache:
        cfg.data.use_cache = False
    return cfg


def print_summary(m: dict, name: str) -> None:
    """Console summary — the same numbers that head the HTML report."""
    line = "-" * 62
    print(f"\n{line}\n  BACKTEST SUMMARY - {name}\n{line}")
    rows = [
        ("Period", f"{m['start_date'][:16]}  ->  {m['end_date'][:16]}"),
        ("Trading sessions", f"{m['trading_sessions']:,}"),
        ("Starting capital", f"INR {m['starting_capital']:,.0f}"),
        ("Final equity", f"INR {m['final_equity']:,.0f}"),
        ("Net profit", f"INR {m['net_profit']:,.0f}"),
        ("Total return", f"{m['total_return_pct']:.2f}%"),
        ("CAGR", f"{m['cagr_pct']:.2f}%"),
        ("Max drawdown", f"{m['max_drawdown_pct']:.2f}%"),
        ("Sharpe / Sortino", f"{m['sharpe']:.2f} / {m['sortino']:.2f}"),
        ("Calmar", f"{m['calmar']:.2f}"),
        ("", ""),
        ("Total trades", f"{m['total_trades']:,}"),
        ("Win rate", f"{m['win_rate_pct']:.1f}%"),
        ("Profit factor", f"{m['profit_factor']:.2f}"),
        ("Expectancy", f"INR {m['expectancy']:,.0f}  ({m['expectancy_r']:.2f} R)"),
        ("Avg win / avg loss",
         f"INR {m['avg_win']:,.0f} / INR {m['avg_loss']:,.0f}"),
        ("Payoff ratio", f"{m['payoff_ratio']:.2f}"),
        ("Max consecutive losses", f"{m['max_consecutive_losses']}"),
        ("Long / short trades", f"{m['long_trades']} / {m['short_trades']}"),
        ("Transaction costs paid", f"INR {m['total_costs']:,.0f}"),
    ]
    for k, v in rows:
        print(f"  {k:<26}{v}" if k else "")
    print(line)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    cfg = apply_overrides(Config.from_yaml(args.config), args)
    tag = args.tag or Path(args.config).stem + "_" + cfg.data.interval

    # 1. Data
    raw = load_ohlcv(cfg.data)
    intraday = is_intraday_interval(cfg.data.interval)
    bars = filter_session(raw, cfg.session, intraday)
    if len(bars) < 200:
        logging.warning(
            "Only %d bars after session filtering — indicators need a warm-up "
            "period, so results will be thin.", len(bars)
        )
    logging.info("Loaded %d bars: %s -> %s", len(bars), bars.index[0], bars.index[-1])

    # 2. Indicators and signals
    prepared = prepare(bars, cfg, intraday)

    # 3. Simulation
    result = run_backtest(prepared, cfg, intraday)

    # 4. Output
    m = mx.compute_metrics(result.trades, result.equity_curve,
                           cfg.risk.starting_capital)
    print_summary(m, f"{cfg.data.name} {cfg.data.interval}")

    if not args.no_report:
        title = f"{cfg.data.name} — Intraday Trend-Following Backtest ({cfg.data.interval})"
        paths = write_report(result, tag, title)
        print("\n  Files written:")
        for kind, p in paths.items():
            print(f"    {kind:<8} {p}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
