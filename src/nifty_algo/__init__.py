"""
nifty-trend-algo — intraday trend-following for Nifty 50 and Nifty Bank.

Package layout
    config.py      every tunable parameter, loaded from YAML
    data.py        Yahoo Finance / CSV loading, session filtering, caching
    indicators.py  EMA, ATR, ADX, Supertrend, Donchian — all causal
    strategy.py    the trading rules, and only the trading rules
    backtest.py    event-driven simulator with realistic fills and costs
    metrics.py     performance statistics
    report.py      self-contained HTML report + CSV exports

Typical use is via the command line — see run_backtest.py — but the pieces are
importable if you want to drive them yourself:

    from nifty_algo import Config, load_ohlcv, prepare, run_backtest
"""

from .config import (  # noqa: F401
    BacktestConfig,
    Config,
    CostConfig,
    DataConfig,
    RiskConfig,
    SessionConfig,
    StrategyConfig,
)
from .data import (  # noqa: F401
    filter_session,
    is_intraday_interval,
    load_ohlcv,
)
from .strategy import prepare  # noqa: F401
from .backtest import BacktestResult, run_backtest  # noqa: F401
from .metrics import compute_metrics  # noqa: F401
from .report import write_report  # noqa: F401

__version__ = "1.0.0"

__all__ = [
    "Config", "DataConfig", "SessionConfig", "StrategyConfig",
    "RiskConfig", "CostConfig", "BacktestConfig",
    "load_ohlcv", "filter_session", "is_intraday_interval",
    "prepare", "run_backtest", "BacktestResult",
    "compute_metrics", "write_report",
    "__version__",
]
