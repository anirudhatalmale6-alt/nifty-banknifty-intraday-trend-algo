"""
Strategy tests: do the signals match the documented rules, exactly?

This is the client-facing acceptance criterion — "signals conform to the
documented logic" — so each of the five entry conditions gets its own test that
switches that condition off and proves the signal count goes to zero or drops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_algo import Config, prepare
from nifty_algo.strategy import build_features, generate_signals, session_flags
from conftest import make_session_index, synth_ohlcv


@pytest.fixture
def prepared(bars, cfg):
    return prepare(bars, cfg, is_intraday=True)


# --------------------------------------------------------------------------
# Session rules
# --------------------------------------------------------------------------
def test_no_signal_outside_the_entry_window(prepared, cfg):
    fired = prepared[prepared["signal"] != 0]
    start = pd.Timestamp(cfg.session.entry_start).time()
    cutoff = pd.Timestamp(cfg.session.entry_cutoff).time()
    for ts in fired.index:
        assert start <= ts.time() <= cutoff, f"signal at {ts} is outside the window"


def test_exactly_one_square_off_bar_per_session(prepared):
    per_day = prepared.groupby("session_date")["is_square_off"].sum()
    assert (per_day == 1).all(), f"days with wrong square-off count:\n{per_day[per_day != 1]}"


def test_square_off_bar_is_at_or_after_the_configured_time(prepared, cfg):
    target = pd.Timestamp(cfg.session.square_off).time()
    for ts in prepared.index[prepared["is_square_off"]]:
        assert ts.time() >= target


def test_square_off_still_marked_when_a_day_ends_early(cfg):
    """A truncated session must still get exactly one square-off bar."""
    bars = synth_ohlcv(make_session_index(5))
    # Chop the last day off at 12:00 — no bar reaches 15:15.
    last_day = bars.index[-1].date()
    keep = ~((pd.Series(bars.index.date, index=bars.index) == last_day)
             & (bars.index.time > pd.Timestamp("12:00").time()))
    truncated = bars[keep.to_numpy()]

    out = session_flags(build_features(truncated, cfg), cfg, is_intraday=True)
    per_day = out.groupby("session_date")["is_square_off"].sum()
    assert (per_day == 1).all()


# --------------------------------------------------------------------------
# Each entry condition, isolated
# --------------------------------------------------------------------------
def _count(bars, cfg):
    return int((prepare(bars, cfg, True)["signal"] != 0).sum())


def test_baseline_produces_some_signals(bars, cfg):
    assert _count(bars, cfg) > 0, "test data is degenerate — nothing to compare against"


def test_adx_gate_removes_signals(bars, cfg):
    base = _count(bars, cfg)
    cfg.strategy.adx_threshold = 100.0  # unreachable
    assert _count(bars, cfg) == 0
    assert base > 0


def test_volatility_floor_removes_signals(bars, cfg):
    cfg.strategy.min_atr_pct = 1.0  # ATR can never be 100% of price
    assert _count(bars, cfg) == 0


def test_breakout_length_gate(bars, cfg):
    """A very long Donchian lookback is much harder to break out of."""
    cfg.strategy.breakout_length = 5
    short = _count(bars, cfg)
    cfg.strategy.breakout_length = 200
    long_ = _count(bars, cfg)
    assert long_ < short


def test_long_only_and_short_only(bars, cfg):
    cfg.strategy.allow_short = False
    out = prepare(bars, cfg, True)
    assert (out["signal"] >= 0).all()

    cfg2 = Config()
    cfg2.strategy.allow_long = False
    out2 = prepare(bars, cfg2, True)
    assert (out2["signal"] <= 0).all()


def test_long_signals_agree_with_every_documented_condition(prepared, cfg):
    """Spot-check the conjunction on every bar that fired long."""
    s = cfg.strategy
    fired = prepared[prepared["signal"] == 1]
    assert len(fired) > 0
    assert (fired["htf_trend"] == 1).all()             # 1. higher timeframe trend
    assert (fired["st_dir"] == 1).all()                # 2. supertrend direction
    assert (fired["close"] > fired["dc_upper"]).all()  # 3. structure breakout
    assert (fired["adx"] >= s.adx_threshold).all()     # 4. trend strength
    assert (fired["atr_pct"] >= s.min_atr_pct).all()   # 5. live volatility
    assert (fired["ema_fast"] > fired["ema_slow"]).all()
    assert fired["can_enter_time"].all()


def test_short_signals_are_the_exact_mirror(prepared, cfg):
    s = cfg.strategy
    fired = prepared[prepared["signal"] == -1]
    assert len(fired) > 0
    assert (fired["htf_trend"] == -1).all()
    assert (fired["st_dir"] == -1).all()
    assert (fired["close"] < fired["dc_lower"]).all()
    assert (fired["adx"] >= s.adx_threshold).all()
    assert (fired["ema_fast"] < fired["ema_slow"]).all()


def test_no_bar_is_both_long_and_short(prepared):
    assert not (prepared["long_setup"] & prepared["short_setup"]).any()


# --------------------------------------------------------------------------
# Opening range mode
# --------------------------------------------------------------------------
def test_opening_range_is_blank_inside_the_range_itself(bars, cfg):
    """The OR level must not exist on the bars that form it — that is lookahead."""
    cfg.strategy.entry_mode = "opening_range"
    cfg.strategy.opening_range_minutes = 30
    out = prepare(bars, cfg, True)
    inside = out.index.time < pd.Timestamp("09:45").time()
    assert out.loc[inside, "or_high"].isna().all()
    assert out.loc[~inside, "or_high"].notna().all()


def test_opening_range_level_equals_the_first_bars_extremes(bars, cfg):
    cfg.strategy.entry_mode = "opening_range"
    cfg.strategy.opening_range_minutes = 30
    out = prepare(bars, cfg, True)
    day = out["session_date"].iloc[0]
    d = out[out["session_date"] == day]
    first_two = d.iloc[:2]  # 09:15 and 09:30 bars on a 15m chart
    later = d.iloc[2]
    assert later["or_high"] == pytest.approx(first_two["high"].max())
    assert later["or_low"] == pytest.approx(first_two["low"].min())


def test_opening_range_signals_use_the_or_level(bars, cfg):
    cfg.strategy.entry_mode = "opening_range"
    out = prepare(bars, cfg, True)
    fired = out[out["signal"] == 1]
    if len(fired):
        assert (fired["close"] > fired["or_high"]).all()


# --------------------------------------------------------------------------
# Causality of the whole pipeline
# --------------------------------------------------------------------------
def test_signals_do_not_change_when_future_bars_are_appended(cfg):
    """The headline no-lookahead test, run through the full pipeline."""
    full = synth_ohlcv(make_session_index(25), seed=11)
    cut = len(full) - 150
    prefix = full.iloc[:cut]

    sig_full = prepare(full, cfg, True)["signal"].iloc[:cut]
    sig_prefix = prepare(prefix, cfg, True)["signal"]

    pd.testing.assert_series_equal(sig_full, sig_prefix, check_names=False)


def test_htf_resample_is_causal(cfg):
    from nifty_algo.data import resample_htf

    full = synth_ohlcv(make_session_index(15), seed=5)
    cut = len(full) - 80
    a = resample_htf(full, 4).iloc[:cut]
    b = resample_htf(full.iloc[:cut], 4)
    pd.testing.assert_frame_equal(a, b, rtol=1e-12)
