"""
Technical indicators used by the trend-following strategy.

Every function in this module is *causal*: the value returned at bar `i` is
computed only from bars `0..i`. Nothing here can peek at future bars, which is
what keeps the backtest honest. If you add an indicator, keep that property.

All functions take/return pandas Series or DataFrames indexed by timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average.

    `adjust=False` gives the classic recursive EMA that TradingView's ta.ema()
    uses, so the Python and Pine Script versions produce identical numbers.
    """
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(length, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA / SMMA).

    Wilder used alpha = 1/length rather than 2/(length+1). ATR, ADX and RSI are
    all defined with this smoother, so getting it right matters if you want the
    numbers to line up with TradingView / your broker's charts.
    """
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range: the largest of (H-L), |H-prev C|, |L-prev C|."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range (Wilder-smoothed). Used for stops and position size."""
    return rma(true_range(df), length)


# --------------------------------------------------------------------------
# Trend strength
# --------------------------------------------------------------------------
def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Average Directional Index plus the +DI / -DI components.

    ADX measures how *strongly* the market is trending, regardless of direction.
    We use it purely as a gate: below the threshold the market is chopping and
    the strategy stands aside. Returns a DataFrame with columns adx, plus_di,
    minus_di.
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    # Directional movement: only the dominant side counts on any given bar.
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr_ = rma(true_range(df), length)
    # Guard against a zero ATR on completely flat synthetic data.
    atr_safe = atr_.replace(0.0, np.nan)

    plus_di = 100.0 * rma(plus_dm, length) / atr_safe
    minus_di = 100.0 * rma(minus_dm, length) / atr_safe

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_ = rma(dx, length)

    return pd.DataFrame(
        {"adx": adx_, "plus_di": plus_di, "minus_di": minus_di}, index=df.index
    )


# --------------------------------------------------------------------------
# Trend direction / structure
# --------------------------------------------------------------------------
def supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend: an ATR band that flips side when price closes through it.

    This is the strategy's primary trend-direction signal and its trailing exit.
    Returns a DataFrame with:
        st_line  - the active band (support when long, resistance when short)
        st_dir   - +1 when the trend is up, -1 when it is down

    The recursive part (the band "ratchets" and only loosens on a flip) has to be
    done in a loop; it cannot be vectorised without changing the definition.
    """
    hl2 = (df["high"] + df["low"]) / 2.0
    atr_ = atr(df, length)

    upper_basic = hl2 + multiplier * atr_
    lower_basic = hl2 - multiplier * atr_

    close = df["close"].to_numpy(dtype=float)
    ub = upper_basic.to_numpy(dtype=float)
    lb = lower_basic.to_numpy(dtype=float)
    n = len(df)

    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    # Find the first bar where ATR is available; the recursion starts there.
    valid = np.where(~np.isnan(ub) & ~np.isnan(lb))[0]
    if len(valid) == 0:
        return pd.DataFrame(
            {"st_line": np.nan, "st_dir": np.nan}, index=df.index
        )
    start = int(valid[0])

    final_ub[start] = ub[start]
    final_lb[start] = lb[start]
    direction[start] = 1.0 if close[start] >= lb[start] else -1.0

    for i in range(start + 1, n):
        # The upper band only ratchets DOWN while price stays below it.
        if ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        # The lower band only ratchets UP while price stays above it.
        if lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        # Direction flips only on a close through the *active* band.
        if direction[i - 1] == 1.0:
            direction[i] = -1.0 if close[i] < final_lb[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] > final_ub[i] else -1.0

    line = np.where(direction == 1.0, final_lb, final_ub)

    return pd.DataFrame({"st_line": line, "st_dir": direction}, index=df.index)


def donchian(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Donchian channel of the *previous* `length` bars.

    Deliberately shifted by one bar so that `dc_upper` at bar i is the highest
    high of bars i-length .. i-1. That is the level price must break to trigger
    an entry — including the current bar's own high would be lookahead.
    """
    upper = df["high"].rolling(length, min_periods=length).max().shift(1)
    lower = df["low"].rolling(length, min_periods=length).min().shift(1)
    mid = (upper + lower) / 2.0
    return pd.DataFrame(
        {"dc_upper": upper, "dc_lower": lower, "dc_mid": mid}, index=df.index
    )
