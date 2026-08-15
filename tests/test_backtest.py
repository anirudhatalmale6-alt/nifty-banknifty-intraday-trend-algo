"""
Backtester tests: execution, risk management and cost accounting.

These are the tests that protect the numbers in the report. A strategy that is
right and a backtester that is wrong produce exactly the same artefact as a
strategy that is wrong — a plausible-looking equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_algo import Config, prepare, run_backtest
from nifty_algo.backtest import _fill_price, _order_cost, _position_size
from conftest import make_session_index, synth_ohlcv


@pytest.fixture
def result(bars, cfg):
    cfg.strategy.adx_threshold = 12.0  # loosen so the synthetic data trades
    return run_backtest(prepare(bars, cfg, True), cfg, True)


# --------------------------------------------------------------------------
# Costs and fills
# --------------------------------------------------------------------------
def test_slippage_always_works_against_you(cfg):
    cfg.costs.slippage_pct = 0.001
    assert _fill_price(100.0, is_buy=True, cfg=cfg) == pytest.approx(100.1)
    assert _fill_price(100.0, is_buy=False, cfg=cfg) == pytest.approx(99.9)


def test_order_cost_matches_a_hand_calculation(cfg):
    """Sell side of one Nifty lot at 24,000 — worked out by hand."""
    c = cfg.costs
    price, qty = 24000.0, 75
    turnover = price * qty  # 18,00,000

    brokerage = 20.0
    exchange = turnover * c.exchange_txn_pct
    sebi = turnover * c.sebi_fee_pct
    stt = turnover * c.stt_sell_pct
    gst = (brokerage + exchange + sebi) * c.gst_pct
    expected = brokerage + exchange + sebi + stt + gst

    assert _order_cost(price, qty, is_buy=False, cfg=cfg) == pytest.approx(expected)


def test_stt_is_charged_on_sells_and_stamp_duty_on_buys(cfg):
    buy = _order_cost(24000.0, 75, is_buy=True, cfg=cfg)
    sell = _order_cost(24000.0, 75, is_buy=False, cfg=cfg)
    # STT (0.02%) dwarfs stamp duty (0.002%), so the sell side must cost more.
    assert sell > buy


def test_costs_scale_with_turnover(cfg):
    one = _order_cost(24000.0, 75, True, cfg)
    two = _order_cost(24000.0, 150, True, cfg)
    # Not exactly 2x — the flat brokerage does not scale.
    assert two > one
    assert two < 2 * one


# --------------------------------------------------------------------------
# Position sizing
# --------------------------------------------------------------------------
def test_size_respects_the_risk_per_trade(cfg):
    cfg.risk.round_to_lots = False
    cfg.risk.risk_per_trade_pct = 0.01
    qty, risk = _position_size(1_000_000.0, 24000.0, 23900.0, cfg)
    # Stop is 100 points away; 1% of 10 lakh is 10,000 -> 100 units.
    assert qty == 100
    assert risk == pytest.approx(10_000.0)


def test_size_is_rounded_down_to_whole_lots(cfg):
    cfg.risk.round_to_lots = True
    cfg.risk.lot_size = 75
    qty, _ = _position_size(1_000_000.0, 24000.0, 23950.0, cfg)
    # 10,000 / 50 = 200 units -> 2 lots -> 150. Never rounded up.
    assert qty == 150


def test_leverage_cap_binds(cfg):
    cfg.risk.round_to_lots = False
    cfg.risk.max_leverage = 1.0
    qty, _ = _position_size(1_000_000.0, 24000.0, 23999.0, cfg)
    # A 1-point stop would otherwise ask for 10,000 units. Leverage caps it at
    # 1,000,000 / 24,000 = 41.
    assert qty == 41


def test_one_lot_fallback_respects_the_risk_ceiling(cfg):
    cfg.risk.lot_size = 35
    cfg.risk.allow_min_one_lot = True
    cfg.risk.max_risk_per_trade_pct = 0.03

    # Stop 500 points wide: one lot risks 35 * 500 = 17,500 = 1.75% -> allowed.
    qty, risk = _position_size(1_000_000.0, 52000.0, 51500.0, cfg)
    assert qty == 35
    assert risk == pytest.approx(17_500.0)

    # Stop 1,200 points wide: one lot risks 42,000 = 4.2% -> refused.
    qty2, _ = _position_size(1_000_000.0, 52000.0, 50800.0, cfg)
    assert qty2 == 0


def test_one_lot_fallback_can_be_switched_off(cfg):
    cfg.risk.lot_size = 35
    cfg.risk.allow_min_one_lot = False
    qty, _ = _position_size(1_000_000.0, 52000.0, 51500.0, cfg)
    assert qty == 0


def test_zero_or_negative_stop_distance_is_refused(cfg):
    assert _position_size(1_000_000.0, 24000.0, 24000.0, cfg) == (0, 0.0)
    assert _position_size(1_000_000.0, 24000.0, float("nan"), cfg) == (0, 0.0)


# --------------------------------------------------------------------------
# Execution rules
# --------------------------------------------------------------------------
def test_it_actually_trades(result):
    assert len(result.trades) > 0, "no trades — the rest of these tests prove nothing"


def test_entries_fill_on_the_bar_after_the_signal(bars, cfg):
    cfg.strategy.adx_threshold = 12.0
    prepared = prepare(bars, cfg, True)
    res = run_backtest(prepared, cfg, True)
    idx = list(prepared.index)
    for _, t in res.trades.iterrows():
        i = idx.index(t["entry_time"])
        assert i > 0
        # The bar BEFORE the entry must have carried the signal, in the same
        # direction, and must not itself be the fill bar.
        assert prepared["signal"].iloc[i - 1] == t["direction"]


def test_never_more_than_one_position_at_a_time(result):
    t = result.trades.sort_values("entry_time")
    for prev, nxt in zip(t.itertuples(), t.iloc[1:].itertuples()):
        assert nxt.entry_time >= prev.exit_time


def test_no_position_survives_the_close(result, cfg):
    """Every trade must be flat by the square-off time, same day it opened."""
    limit = pd.Timestamp(cfg.session.square_off).time()
    for _, t in result.trades.iterrows():
        assert t["exit_time"].date() == t["entry_time"].date(), "overnight position"
        assert t["exit_time"].time() <= limit


def test_max_trades_per_day_is_enforced(bars, cfg):
    cfg.strategy.adx_threshold = 8.0   # let a lot of signals through
    cfg.risk.max_trades_per_day = 1
    res = run_backtest(prepare(bars, cfg, True), cfg, True)
    per_day = res.trades.groupby(res.trades["entry_time"].dt.date).size()
    assert (per_day <= 1).all()


def test_daily_loss_limit_stops_the_day(bars, cfg):
    cfg.strategy.adx_threshold = 8.0
    cfg.risk.max_trades_per_day = 20
    cfg.risk.daily_loss_limit_pct = 0.0001  # trip on the first loss
    res = run_backtest(prepare(bars, cfg, True), cfg, True)
    assert res.skipped["daily_loss_limit"] > 0

    # Once a day's cumulative realised P&L goes negative the limit has tripped,
    # so no further position may be OPENED that day. The trade that tripped it
    # is of course allowed to have happened.
    for _, day_trades in res.trades.groupby(res.trades["entry_time"].dt.date):
        day_trades = day_trades.sort_values("exit_time")
        cum = day_trades["net_pnl"].cumsum().to_numpy()
        breached = np.flatnonzero(cum < 0)
        if len(breached):
            first = int(breached[0])
            # Anything opened after that trade closed is a violation.
            trip_time = day_trades["exit_time"].iloc[first]
            later = day_trades[day_trades["entry_time"] > trip_time]
            assert later.empty, f"opened {len(later)} trade(s) after the daily limit tripped"


def test_daily_loss_limit_off_by_default_lets_the_day_continue(bars, cfg):
    cfg.strategy.adx_threshold = 8.0
    cfg.risk.max_trades_per_day = 20
    cfg.risk.daily_loss_limit_pct = 0.0  # disabled
    res = run_backtest(prepare(bars, cfg, True), cfg, True)
    assert res.skipped["daily_loss_limit"] == 0


def test_stop_loss_is_never_filled_better_than_the_stop(result, cfg):
    """A stop exit must fill at the stop price or worse — never better.

    'Worse' means below the stop for a long, above it for a short, by exactly
    the slippage assumption. Compared against final_stop, not initial_stop,
    because the break-even move and the trail both legitimately move it.
    """
    slip = cfg.costs.slippage_pct
    stops = result.trades[result.trades["exit_reason"] == "stop"]
    assert len(stops) > 0
    for _, t in stops.iterrows():
        if t["direction"] == 1:
            # Long exits by selling: fill = stop * (1 - slippage).
            assert t["exit_price"] == pytest.approx(t["final_stop"] * (1 - slip), rel=1e-9)
            assert t["exit_price"] <= t["final_stop"]
        else:
            assert t["exit_price"] == pytest.approx(t["final_stop"] * (1 + slip), rel=1e-9)
            assert t["exit_price"] >= t["final_stop"]


def test_trailed_stop_only_ever_moves_in_your_favour(bars, cfg):
    """final_stop must never be worse than initial_stop."""
    cfg.strategy.adx_threshold = 12.0
    cfg.strategy.supertrend_trail_mode = "intrabar"
    res = run_backtest(prepare(bars, cfg, True), cfg, True)
    longs = res.trades[res.trades["direction"] == 1]
    shorts = res.trades[res.trades["direction"] == -1]
    assert (longs["final_stop"] >= longs["initial_stop"] - 1e-9).all()
    assert (shorts["final_stop"] <= shorts["initial_stop"] + 1e-9).all()


def test_initial_stop_is_the_configured_atr_distance(bars, cfg):
    cfg.strategy.adx_threshold = 12.0
    cfg.strategy.breakeven_atr_multiplier = 0.0
    prepared = prepare(bars, cfg, True)
    res = run_backtest(prepared, cfg, True)
    idx = list(prepared.index)
    for _, t in res.trades.iterrows():
        i = idx.index(t["entry_time"])
        atr_ref = prepared["atr"].iloc[i - 1]
        expected = t["entry_price"] - t["direction"] * cfg.strategy.stop_atr_multiplier * atr_ref
        assert t["initial_stop"] == pytest.approx(expected, rel=1e-9)


def test_net_pnl_equals_gross_minus_costs(result):
    d = result.trades
    assert np.allclose(d["net_pnl"], d["gross_pnl"] - d["costs"])


def test_equity_curve_ends_where_the_trades_say_it_should(result, cfg):
    expected = cfg.risk.starting_capital + result.trades["net_pnl"].sum()
    assert result.equity_curve["equity"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_equity_curve_covers_every_bar(result, bars):
    assert len(result.equity_curve) == len(bars)


def test_long_only_produces_no_shorts(bars, cfg):
    cfg.strategy.adx_threshold = 12.0
    cfg.strategy.allow_short = False
    res = run_backtest(prepare(bars, cfg, True), cfg, True)
    assert len(res.trades) > 0
    assert (res.trades["direction"] == 1).all()


# --------------------------------------------------------------------------
# Reproducibility — the client's third acceptance criterion
# --------------------------------------------------------------------------
def test_two_identical_runs_give_identical_results(bars, cfg):
    a = run_backtest(prepare(bars, cfg, True), cfg, True)
    b = run_backtest(prepare(bars, cfg, True), cfg, True)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.equity_curve, b.equity_curve)


def test_results_are_unchanged_by_data_that_arrives_later(cfg):
    """Backtesting a prefix must reproduce the prefix of the full backtest.

    This is the strongest single guarantee against lookahead: if any part of the
    system saw the future, truncating the data would change the past.
    """
    cfg.strategy.adx_threshold = 12.0
    full = synth_ohlcv(make_session_index(24), seed=17)

    # Cut at a session boundary so the comparison is clean.
    days = sorted(set(full.index.date))
    cut_day = days[-6]
    prefix = full[full.index.date < cut_day]

    res_full = run_backtest(prepare(full, cfg, True), cfg, True)
    res_pre = run_backtest(prepare(prefix, cfg, True), cfg, True)

    a = res_full.trades[res_full.trades["exit_time"] < prefix.index[-1]].reset_index(drop=True)
    b = res_pre.trades[res_pre.trades["exit_reason"] != "end_of_data"].reset_index(drop=True)

    assert len(a) == len(b) and len(a) > 0
    pd.testing.assert_series_equal(a["entry_time"], b["entry_time"])
    pd.testing.assert_series_equal(a["net_pnl"], b["net_pnl"], rtol=1e-9)


# --------------------------------------------------------------------------
# Daily-bar mode
# --------------------------------------------------------------------------
def test_daily_bars_do_not_square_off_or_reset(cfg):
    """On daily data the intraday machinery must switch itself off."""
    idx = pd.date_range("2020-01-01", periods=800, freq="B", tz="Asia/Kolkata")
    daily = synth_ohlcv(idx, seed=23)
    cfg.data.interval = "1d"
    cfg.strategy.adx_threshold = 12.0

    res = run_backtest(prepare(daily, cfg, False), cfg, is_intraday=False)
    assert len(res.trades) > 0
    assert "square_off" not in set(res.trades["exit_reason"])
    # Positions are allowed to span more than one bar here.
    assert res.trades["bars_held"].max() > 1
