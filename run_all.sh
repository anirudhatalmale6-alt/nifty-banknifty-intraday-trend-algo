#!/usr/bin/env bash
# Regenerate every report in output/ from scratch.
#
#   ./run_all.sh
#
# The first run downloads from Yahoo Finance and caches to data/cache/.
# Every run after that reads the cache, so the reports are reproducible.
set -euo pipefail
cd "$(dirname "$0")"

# Use whichever Python is on the PATH.
PY="${PYTHON:-$(command -v python3 || command -v python)}"

echo "==> Tests"
"$PY" -m pytest tests/ -q

echo
echo "==> Primary backtests (15-minute signal timeframe)"
"$PY" run_backtest.py --config config/nifty.yaml     --tag nifty_15m
"$PY" run_backtest.py --config config/banknifty.yaml --tag banknifty_15m

echo
echo "==> Faster timeframe, for the cost comparison in README section 7"
"$PY" run_backtest.py --config config/nifty.yaml     --interval 5m --tag nifty_5m
"$PY" run_backtest.py --config config/banknifty.yaml --interval 5m --tag banknifty_5m

echo
echo "==> Long-sample validation on daily bars (no intraday square-off)"
"$PY" run_backtest.py --config config/nifty.yaml \
    --interval 1d --period 15y --tag nifty_daily_15y
"$PY" run_backtest.py --config config/banknifty.yaml \
    --interval 1d --period 15y --tag banknifty_daily_15y

echo
echo "==> Cost sensitivity"
"$PY" tools/sensitivity.py costs --config config/nifty.yaml     --tag nifty_15m
"$PY" tools/sensitivity.py costs --config config/banknifty.yaml --tag banknifty_15m

echo
echo "==> Out-of-sample split"
"$PY" tools/sensitivity.py oos --config config/nifty.yaml --tag nifty_15m

echo
echo "Done. Open output/nifty_15m_report.html in a browser."
