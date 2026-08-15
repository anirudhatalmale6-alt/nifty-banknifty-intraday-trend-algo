"""
The strategy: intraday trend-following on Nifty 50 / Nifty Bank.

WHAT IT DOES, IN PLAIN ENGLISH
------------------------------
It is a breakout system with a trend filter in front of it. There is no mean
reversion anywhere in this file. It never buys weakness or sells strength — it
only joins a move that is already underway and already confirmed on a slower
timeframe.

An entry needs FIVE things to agree at the close of a bar:

  1. HIGHER TIMEFRAME TREND   The slower timeframe (signal TF x htf_multiplier)
                              must be trending the same way: fast EMA above slow
                              EMA, and the slow EMA actually rising. This is the
                              "which way is the river flowing" test.

  2. SUPERTREND DIRECTION     The signal timeframe Supertrend must already have
                              flipped to that side. This is the trend's own
                              statement about itself, and it doubles as the
                              trailing stop later.

  3. STRUCTURE BREAKOUT       Price must close beyond the highest high (or lowest
                              low) of the previous `breakout_length` bars. This
                              is the trigger — without it the system would enter
                              in the middle of a range that merely looks trendy.

  4. TREND STRENGTH           ADX must be at or above `adx_threshold`. ADX does
                              not care about direction, only conviction. Below
                              the threshold the market is chopping and breakouts
                              fail; the system stands aside.

  5. LIVE VOLATILITY          ATR as a fraction of price must clear `min_atr_pct`.
                              In a dead tape the stop distance collapses, the
                              position size explodes, and one tick of noise
                              stops you out. This filter refuses those setups.

Plus the session rules: nothing before `entry_start` (opening auction noise),
nothing after `entry_cutoff` (no time left for the trade to work), everything
flat by `square_off`.

Short entries are the exact mirror image. Nothing is asymmetric.

EXITS (handled in backtest.py, driven by the columns produced here)
  * Initial stop at entry -/+ stop_atr_multiplier x ATR
  * Stop moves to break-even once price runs breakeven_atr_multiplier x ATR your way
  * Supertrend line then trails the stop for the rest of the move
  * Optional fixed target at target_atr_multiplier x ATR (off by default — a
    trend system makes its money on the tail, so capping the tail is expensive)
  * EMA cross back against the position closes it
  * Hard square-off at `square_off`

WHY THESE INDICATORS
  EMA/Supertrend/Donchian/ADX/ATR are all causal, all standard, and all available
  natively in Pine Script — which is what lets pine/nifty_trend.pine reproduce
  these signals bar for bar on your TradingView chart.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .data import resample_htf
from . import indicators as ind

log = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach every indicator the strategy needs to the price frame.

    Returns a copy of `df` with the indicator columns added. Purely a function
    of past and present bars — see indicators.py for the causality guarantee.
    """
    s = cfg.strategy
    out = df.copy()

    # --- Signal timeframe -------------------------------------------------
    out["ema_fast"] = ind.ema(out["close"], s.ema_fast)
    out["ema_slow"] = ind.ema(out["close"], s.ema_slow)
    out["atr"] = ind.atr(out, s.atr_length)
    out["atr_pct"] = out["atr"] / out["close"]

    adx_df = ind.adx(out, s.adx_length)
    out["adx"] = adx_df["adx"]
    out["plus_di"] = adx_df["plus_di"]
    out["minus_di"] = adx_df["minus_di"]

    st = ind.supertrend(out, s.supertrend_length, s.supertrend_multiplier)
    out["st_line"] = st["st_line"]
    out["st_dir"] = st["st_dir"]

    dc = ind.donchian(out, s.breakout_length)
    out["dc_upper"] = dc["dc_upper"]
    out["dc_lower"] = dc["dc_lower"]

    # --- Higher timeframe trend filter ------------------------------------
    htf = resample_htf(out[["open", "high", "low", "close", "volume"]],
                       s.htf_multiplier)
    out["htf_ema_fast"] = ind.ema(htf["close"], s.htf_ema_fast)
    out["htf_ema_slow"] = ind.ema(htf["close"], s.htf_ema_slow)

    # "Rising" = the slow EMA is above where it was `slope_lookback` bars ago.
    slope = out["htf_ema_slow"] - out["htf_ema_slow"].shift(s.htf_slope_lookback)
    out["htf_slope"] = slope

    out["htf_trend"] = np.select(
        [
            (out["htf_ema_fast"] > out["htf_ema_slow"]) & (slope > 0),
            (out["htf_ema_fast"] < out["htf_ema_slow"]) & (slope < 0),
        ],
        [1, -1],
        default=0,
    )

    return out


def session_flags(df: pd.DataFrame, cfg: Config, is_intraday: bool) -> pd.DataFrame:
    """Add the time-of-day gates: can we enter, must we square off."""
    out = df.copy()

    if not is_intraday:
        # Daily bars: no intraday session concept. Every bar is tradeable and
        # nothing is force-closed by the clock.
        out["can_enter_time"] = True
        out["is_square_off"] = False
        out["session_date"] = out.index.date
        out["or_high"] = np.nan
        out["or_low"] = np.nan
        return out

    sess = cfg.session
    t = out.index.time
    entry_start = pd.Timestamp(sess.entry_start).time()
    entry_cutoff = pd.Timestamp(sess.entry_cutoff).time()
    square_off = pd.Timestamp(sess.square_off).time()

    out["can_enter_time"] = (t >= entry_start) & (t <= entry_cutoff)
    out["session_date"] = out.index.date

    # The square-off bar is the first bar at or after `square_off` on each day.
    at_or_after = pd.Series(t >= square_off, index=out.index)
    is_first = at_or_after & ~at_or_after.groupby(out["session_date"]).shift(1, fill_value=False)
    out["is_square_off"] = is_first

    # Safety net: if a day has no bar at/after the square-off time (early close,
    # missing data), force the day's last bar to be the square-off bar.
    last_of_day = ~pd.Series(out["session_date"], index=out.index).duplicated(keep="last")
    has_squareoff = is_first.groupby(out["session_date"]).transform("any")
    out["is_square_off"] = out["is_square_off"] | (last_of_day & ~has_squareoff)

    # --- Opening range ----------------------------------------------------
    # High and low of the first `opening_range_minutes` of each session. Only
    # used when entry_mode is "opening_range", but always computed so the report
    # and the Pine version can plot it.
    or_minutes = cfg.strategy.opening_range_minutes
    or_end = (pd.Timestamp(sess.market_open) + pd.Timedelta(minutes=or_minutes)).time()
    in_or = pd.Series((t >= pd.Timestamp(sess.market_open).time()) & (t < or_end),
                      index=out.index)

    day = out["session_date"]
    or_high = out["high"].where(in_or).groupby(day).transform("max")
    or_low = out["low"].where(in_or).groupby(day).transform("min")

    # Blank the level during the opening range itself. The range is not known
    # until the window has closed, so a bar inside it must not see the final
    # high/low — that would be lookahead of the worst kind.
    out["or_high"] = or_high.where(~in_or)
    out["or_low"] = or_low.where(~in_or)

    if cfg.strategy.entry_mode == "opening_range" and not in_or.any():
        log.warning(
            "entry_mode is 'opening_range' but no bar falls inside the first "
            "%d minutes of the session — check that the bar interval divides "
            "the opening range window.", or_minutes
        )

    return out


def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Produce the raw entry signals.

    Adds:
        long_setup  / short_setup  - all five conditions true on this bar's close
        signal                     - +1 long, -1 short, 0 nothing

    `signal` is an *intent*, not a fill. The backtester decides whether it can
    act on it (position already open? daily loss limit hit? trade count used up?)
    and fills it at the next bar's open.
    """
    s = cfg.strategy
    out = df.copy()

    trend_up = out["htf_trend"] == 1
    trend_dn = out["htf_trend"] == -1

    st_up = out["st_dir"] == 1
    st_dn = out["st_dir"] == -1

    # The entry trigger. Everything else on this bar is a filter; this is the
    # thing that actually fires. Both modes use the same "close beyond the
    # level, by at least `breakout_buffer_atr` x ATR" test.
    if s.entry_mode == "opening_range":
        up_level, dn_level = out["or_high"], out["or_low"]
    else:
        up_level, dn_level = out["dc_upper"], out["dc_lower"]
    out["break_level_up"] = up_level
    out["break_level_dn"] = dn_level

    buffer = s.breakout_buffer_atr * out["atr"]
    breakout_up = out["close"] > (up_level + buffer)
    breakout_dn = out["close"] < (dn_level - buffer)

    strong = out["adx"] >= s.adx_threshold
    alive = out["atr_pct"] >= s.min_atr_pct

    ema_up = out["ema_fast"] > out["ema_slow"]
    ema_dn = out["ema_fast"] < out["ema_slow"]

    out["long_setup"] = (
        trend_up & st_up & breakout_up & strong & alive & ema_up
        & out["can_enter_time"] & bool(s.allow_long)
    )
    out["short_setup"] = (
        trend_dn & st_dn & breakout_dn & strong & alive & ema_dn
        & out["can_enter_time"] & bool(s.allow_short)
    )

    # A bar cannot be both. If the data ever produced both (it cannot, the trend
    # filter is mutually exclusive) we would rather take nothing than guess.
    both = out["long_setup"] & out["short_setup"]
    if both.any():  # pragma: no cover - defensive
        log.warning("%d bar(s) flagged long and short at once; ignoring them",
                    int(both.sum()))
        out.loc[both, ["long_setup", "short_setup"]] = False

    out["signal"] = np.select(
        [out["long_setup"], out["short_setup"]], [1, -1], default=0
    ).astype(int)

    # Rows where any indicator is still warming up cannot produce a signal.
    warmup_cols = ["atr", "adx", "st_dir", "break_level_up", "htf_ema_slow", "htf_slope"]
    warm = out[warmup_cols].notna().all(axis=1)
    out.loc[~warm, "signal"] = 0
    out["is_warm"] = warm

    return out


def prepare(df: pd.DataFrame, cfg: Config, is_intraday: bool) -> pd.DataFrame:
    """features -> session flags -> signals, in that order."""
    out = build_features(df, cfg)
    out = session_flags(out, cfg, is_intraday)
    out = generate_signals(out, cfg)
    n_long = int((out["signal"] == 1).sum())
    n_short = int((out["signal"] == -1).sum())
    log.info("Signals generated: %d long, %d short over %d bars",
             n_long, n_short, len(out))
    return out
