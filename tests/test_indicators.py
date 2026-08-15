"""
Indicator tests.

Two things are being checked here:
  1. The maths is right — verified against hand-computed values, not against
     another library that could be wrong in the same way.
  2. The indicators are CAUSAL — appending future bars must never change a value
     that was already computed. This is the property that makes the backtest
     trustworthy, and it is the one that silently breaks when someone "optimises"
     an indicator with a centred rolling window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_algo import indicators as ind
from conftest import make_session_index, synth_ohlcv


# --------------------------------------------------------------------------
def test_ema_matches_manual_recursion():
    """EMA with adjust=False is the textbook recursion — check it exactly."""
    s = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0])
    out = ind.ema(s, 3)

    alpha = 2 / (3 + 1)
    # The first `length-1` values are NaN (min_periods), the third is the SMA-
    # free recursive value pandas produces from the very first observation.
    manual = s.iloc[0]
    for v in s.iloc[1:]:
        manual = alpha * v + (1 - alpha) * manual
    assert out.iloc[:2].isna().all()
    assert out.iloc[-1] == pytest.approx(manual, rel=1e-12)


def test_rma_uses_wilder_alpha():
    """Wilder's smoother is alpha = 1/length, NOT 2/(length+1)."""
    s = pd.Series(np.arange(1.0, 21.0))
    out = ind.rma(s, 5)
    alpha = 1 / 5
    manual = s.iloc[0]
    for v in s.iloc[1:]:
        manual = alpha * v + (1 - alpha) * manual
    assert out.iloc[-1] == pytest.approx(manual, rel=1e-12)
    # And it must differ from a same-length EMA, or we have the wrong smoother.
    assert out.iloc[-1] != pytest.approx(ind.ema(s, 5).iloc[-1], rel=1e-6)


def test_true_range_takes_the_largest_of_three():
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [105.0, 102.0, 110.0],
            "low": [99.0, 95.0, 108.0],
            "close": [104.0, 96.0, 109.0],
        }
    )
    tr = ind.true_range(df)
    assert np.isnan(tr.iloc[0]) or tr.iloc[0] == pytest.approx(6.0)  # H-L
    assert tr.iloc[1] == pytest.approx(9.0)   # H-L = 7, |L - prevC| = 9
    assert tr.iloc[2] == pytest.approx(14.0)  # |H - prevC| = 14


def test_atr_is_positive_and_finite(bars):
    a = ind.atr(bars, 14).dropna()
    assert len(a) > 0
    assert (a > 0).all()
    assert np.isfinite(a).all()


def test_adx_bounded_zero_to_hundred(bars):
    out = ind.adx(bars, 14).dropna()
    assert len(out) > 0
    assert out["adx"].between(0, 100).all()
    assert out["plus_di"].between(0, 100).all()
    assert out["minus_di"].between(0, 100).all()


def test_adx_is_high_in_a_clean_trend():
    """A straight-line advance must register as a strong trend."""
    n = 120
    close = np.linspace(100.0, 200.0, n)
    df = pd.DataFrame(
        {
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": np.ones(n),
        }
    )
    out = ind.adx(df, 14)
    assert out["adx"].iloc[-1] > 40
    assert out["plus_di"].iloc[-1] > out["minus_di"].iloc[-1]


def test_supertrend_direction_flips_with_the_trend():
    """Up leg then down leg: the direction must be +1 then -1."""
    up = np.linspace(100.0, 160.0, 80)
    down = np.linspace(160.0, 100.0, 80)
    close = np.concatenate([up, down])
    df = pd.DataFrame(
        {
            "open": close, "high": close + 0.6, "low": close - 0.6,
            "close": close, "volume": np.ones(len(close)),
        }
    )
    st = ind.supertrend(df, 10, 3.0)
    assert st["st_dir"].iloc[75] == 1
    assert st["st_dir"].iloc[-1] == -1
    # The band sits below price in an uptrend and above it in a downtrend.
    assert st["st_line"].iloc[75] < df["close"].iloc[75]
    assert st["st_line"].iloc[-1] > df["close"].iloc[-1]


def test_supertrend_band_never_loosens_while_the_trend_holds():
    """In an unbroken uptrend the support band may only ratchet up."""
    close = np.linspace(100.0, 200.0, 200)
    df = pd.DataFrame(
        {
            "open": close, "high": close + 0.4, "low": close - 0.4,
            "close": close, "volume": np.ones(200),
        }
    )
    st = ind.supertrend(df, 10, 2.0).dropna()
    up = st[st["st_dir"] == 1]["st_line"]
    assert (up.diff().dropna() >= -1e-9).all()


def test_donchian_excludes_the_current_bar():
    """dc_upper at bar i must be the high of bars i-length .. i-1, not i."""
    highs = [10, 11, 12, 50, 13, 14]
    df = pd.DataFrame(
        {
            "open": highs, "high": highs, "low": [h - 1 for h in highs],
            "close": highs, "volume": [1] * 6,
        }
    ).astype(float)
    dc = ind.donchian(df, 3)
    # Bar 3 has high 50. Its own dc_upper looks at bars 0-2 -> 12.
    assert dc["dc_upper"].iloc[3] == pytest.approx(12.0)
    # Bar 4 looks at bars 1-3, which now includes the 50.
    assert dc["dc_upper"].iloc[4] == pytest.approx(50.0)


# --------------------------------------------------------------------------
# Causality — the property everything else depends on.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn",
    [
        lambda d: ind.ema(d["close"], 20),
        lambda d: ind.atr(d, 14),
        lambda d: ind.adx(d, 14)["adx"],
        lambda d: ind.supertrend(d, 10, 3.0)["st_line"],
        lambda d: ind.supertrend(d, 10, 3.0)["st_dir"],
        lambda d: ind.donchian(d, 20)["dc_upper"],
    ],
    ids=["ema", "atr", "adx", "supertrend_line", "supertrend_dir", "donchian"],
)
def test_indicator_is_causal(fn):
    """Values computed on a prefix must equal values computed on the full set.

    If an indicator peeked ahead, adding 100 future bars would change what it
    said about the past. This test is the tripwire.
    """
    full = synth_ohlcv(make_session_index(20), seed=3)
    cut = len(full) - 100
    prefix = full.iloc[:cut]

    on_full = fn(full).iloc[:cut]
    on_prefix = fn(prefix)

    pd.testing.assert_series_equal(
        on_full, on_prefix, check_names=False, rtol=1e-12, atol=1e-12
    )
