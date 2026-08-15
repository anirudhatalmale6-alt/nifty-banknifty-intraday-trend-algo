"""
Configuration objects.

Everything that can be tuned lives in a YAML file under config/. Nothing in the
strategy or backtester reads a hard-coded number — if you want to change how the
system behaves, change the YAML, not the code.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Where the price data comes from."""

    # Yahoo Finance ticker. ^NSEI = Nifty 50, ^NSEBANK = Nifty Bank.
    symbol: str = "^NSEI"
    # Human-readable name used in the report.
    name: str = "Nifty 50"
    # Bar size the signals are computed on: 5m, 15m, 30m, 1h, 1d ...
    interval: str = "15m"
    # How far back to pull. Yahoo's limits: 60d for intraday <= 30m, 730d for 1h.
    period: str = "60d"
    # Optional explicit window (YYYY-MM-DD). Overrides `period` when both set.
    start: str | None = None
    end: str | None = None
    # Optional CSV to use instead of Yahoo. Must have columns
    # datetime,open,high,low,close,volume. See README for the exact format.
    csv_path: str | None = None
    # Cache downloads to disk so re-runs are byte-for-byte reproducible offline.
    use_cache: bool = True
    cache_dir: str = "data/cache"
    # Exchange timezone. All session times below are interpreted in this zone.
    timezone: str = "Asia/Kolkata"


@dataclass
class SessionConfig:
    """NSE intraday session rules. Times are HH:MM in the exchange timezone."""

    market_open: str = "09:15"
    market_close: str = "15:30"
    # Skip the opening auction noise — no signals before this time.
    entry_start: str = "09:30"
    # Last time a new position may be opened.
    entry_cutoff: str = "14:45"
    # Hard square-off. Any open position is closed at this bar, no exceptions.
    square_off: str = "15:15"


@dataclass
class StrategyConfig:
    """Trend-following parameters.

    The logic in one sentence: trade only in the direction of the higher
    timeframe trend, enter on a breakout of recent structure once trend strength
    confirms, and trail the exit with Supertrend.
    """

    # --- Higher timeframe trend filter -----------------------------------
    # The signal timeframe is resampled up by this factor for the trend read.
    # e.g. 15m bars x 4 = 1h trend. Set to 1 to disable the higher timeframe.
    htf_multiplier: int = 4
    htf_ema_fast: int = 20
    htf_ema_slow: int = 50
    # Trend is "up" only when fast EMA > slow EMA AND the slow EMA is rising.
    htf_slope_lookback: int = 3

    # --- Signal timeframe -------------------------------------------------
    ema_fast: int = 9
    ema_slow: int = 21

    # Supertrend: primary direction signal and the trailing stop.
    supertrend_length: int = 10
    supertrend_multiplier: float = 2.5

    # --- What triggers the entry ------------------------------------------
    #   "donchian"       - break of the highest high / lowest low of the last
    #                      `breakout_length` bars. Works on any timeframe and
    #                      carries structure across sessions.
    #   "opening_range"  - break of the range set in the first
    #                      `opening_range_minutes` of the session. The classic
    #                      Indian intraday trigger (ORB). Rolls fresh each day,
    #                      so entries happen earlier and the trade has more of
    #                      the session left to run before square-off.
    # Both are gated by exactly the same trend filters — only the trigger differs.
    entry_mode: str = "donchian"

    # Donchian breakout length — price must take out this many bars of range.
    breakout_length: int = 20

    # Opening range window, in minutes from the market open. 15 and 30 are the
    # two conventional choices. Only used when entry_mode is "opening_range".
    opening_range_minutes: int = 15
    # Require the breakout bar to close beyond the range by this fraction of
    # ATR, so a one-tick poke through the level does not count as a break.
    breakout_buffer_atr: float = 0.0

    # ADX gate: no entries while trend strength is below this.
    adx_length: int = 14
    adx_threshold: float = 20.0

    # ATR used for stops and sizing.
    atr_length: int = 14
    # Ignore signals when volatility is pathologically low (dead market).
    min_atr_pct: float = 0.0005  # 0.05% of price

    # --- Exits -------------------------------------------------------------
    # Initial stop distance in ATR multiples.
    stop_atr_multiplier: float = 1.5
    # Take profit in ATR multiples. Set to 0 to run a pure trailing exit.
    target_atr_multiplier: float = 0.0
    # Move the stop to break-even once price has gone this many ATR in favour.
    breakeven_atr_multiplier: float = 1.0
    # Trail with the Supertrend line once in profit.
    use_supertrend_trail: bool = True
    # HOW the Supertrend trail is enforced. This one switch matters more than
    # any other parameter in this file:
    #   "close"    - exit when a bar CLOSES through the Supertrend line, filled
    #                at the next bar's open. This is how Supertrend is meant to
    #                be read, and it is what the Pine version plots. Intraday
    #                noise that pokes through the line and recovers does not
    #                take you out.
    #   "intrabar" - the Supertrend line becomes a hard stop order, hit the
    #                moment price touches it. Tighter, locks in more, but on
    #                Nifty/Bank Nifty it converts a lot of eventual winners into
    #                small losses.
    supertrend_trail_mode: str = "close"
    # Also exit if the signal-timeframe EMAs cross back against the position.
    exit_on_ema_cross: bool = True

    # Allow both directions? Set allow_short False for a long-only account.
    allow_long: bool = True
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.supertrend_trail_mode not in ("close", "intrabar"):
            raise ValueError(
                "strategy.supertrend_trail_mode must be 'close' or 'intrabar', "
                f"got {self.supertrend_trail_mode!r}"
            )
        if self.entry_mode not in ("donchian", "opening_range"):
            raise ValueError(
                "strategy.entry_mode must be 'donchian' or 'opening_range', "
                f"got {self.entry_mode!r}"
            )
        if self.htf_multiplier < 1:
            raise ValueError("strategy.htf_multiplier must be >= 1")
        if self.stop_atr_multiplier <= 0:
            raise ValueError("strategy.stop_atr_multiplier must be > 0")


@dataclass
class RiskConfig:
    """Position sizing and the circuit breakers."""

    starting_capital: float = 1_000_000.0  # INR
    # Fraction of *current* equity risked between entry and the initial stop.
    risk_per_trade_pct: float = 0.01  # 1%
    # Cap on notional exposure as a multiple of equity (leverage ceiling).
    max_leverage: float = 5.0
    # Contract size. Nifty futures = 75, Bank Nifty futures = 35 (2026 lot sizes).
    lot_size: int = 75
    # Trade whole lots only. Turn off to backtest a fractional/CFD-style fill.
    round_to_lots: bool = True
    # Derivatives come in indivisible lots, so on a wide stop the risk-based size
    # can round down to zero lots. Two honest ways to handle that:
    #   False - skip the trade. Purest, but on Bank Nifty with a modest account
    #           it silently throws away most of the signals.
    #   True  - take one lot anyway, PROVIDED the resulting risk stays under
    #           max_risk_per_trade_pct. This is what a real trader does, and it
    #           makes the true risk per trade visible instead of hiding it.
    allow_min_one_lot: bool = True
    # Hard ceiling on the risk of any single trade, as a fraction of equity.
    # Applies to the one-lot fallback above: if one lot would risk more than
    # this, the trade is skipped no matter what.
    max_risk_per_trade_pct: float = 0.03  # 3%
    # Hard stop for the day once cumulative P&L hits -X% of the day's opening equity.
    daily_loss_limit_pct: float = 0.02  # 2%
    # Stop taking new trades for the day after this many wins' worth of profit.
    daily_profit_target_pct: float = 0.0  # 0 = disabled
    # Maximum entries per session.
    max_trades_per_day: int = 3
    # Consecutive losing days before the system pauses (0 = disabled).
    max_consecutive_losing_days: int = 0
    # Only one position at a time — this system is not a portfolio strategy.
    max_open_positions: int = 1


@dataclass
class CostConfig:
    """Transaction costs, expressed the way an Indian broker actually charges.

    Defaults are modelled on a discount broker trading *index futures*. If you
    trade options or pay a percentage brokerage, change these — the backtest is
    only as honest as this block.
    """

    # Flat brokerage per executed order (entry and exit are separate orders).
    brokerage_per_order: float = 20.0
    # Exchange transaction charges, as a fraction of turnover.
    exchange_txn_pct: float = 0.0000173  # NSE futures ~0.00173%
    # STT on the SELL side of a futures trade.
    stt_sell_pct: float = 0.0002  # 0.02%
    # SEBI turnover fee.
    sebi_fee_pct: float = 0.000001
    # Stamp duty on the BUY side.
    stamp_duty_pct: float = 0.00002  # 0.002%
    # GST on (brokerage + exchange txn + sebi).
    gst_pct: float = 0.18
    # Slippage assumption, one way, as a fraction of price. 0.0002 = 2 bps.
    # This is the single most important number for realism on an intraday system.
    slippage_pct: float = 0.0002


@dataclass
class BacktestConfig:
    """Execution assumptions for the simulation itself."""

    # Signals are evaluated on bar close and filled at the NEXT bar's open.
    # This is the realistic default. Setting it False fills at the signal bar's
    # close, which flatters the results — only use it to compare with TradingView.
    next_bar_execution: bool = True
    # When both the stop and the target sit inside one bar's range we cannot
    # know which came first. True = assume the stop (pessimistic, recommended).
    pessimistic_intrabar: bool = True
    output_dir: str = "output"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # ---------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a config file. Any key you omit falls back to the default above."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        sections = {
            "data": DataConfig,
            "session": SessionConfig,
            "strategy": StrategyConfig,
            "risk": RiskConfig,
            "costs": CostConfig,
            "backtest": BacktestConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sections.items():
            values = raw.get(key) or {}
            known = {f.name for f in dataclasses.fields(klass)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in config section '{key}': {sorted(unknown)}. "
                    f"Valid keys: {sorted(known)}"
                )
            kwargs[key] = klass(**values)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
