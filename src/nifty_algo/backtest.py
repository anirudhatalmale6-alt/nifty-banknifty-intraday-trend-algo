"""
Event-driven backtester.

Why event-driven and not a vectorised `signal.shift(1) * returns`? Because this
strategy has path-dependent exits — a trailing stop, a break-even move, an
intrabar stop — and a vectorised backtest silently gets those wrong. It is
slower to run and enormously harder to fool.

THE ANTI-LOOKAHEAD CONTRACT
---------------------------
1. A signal is only ever read from a bar that has *closed*.
2. Entries and signal-based exits fill at the NEXT bar's OPEN, with slippage.
3. Stop / target hits are checked against the CURRENT bar's high and low, and
   fill at the stop price (plus slippage) — never at a better price.
4. When a bar's range contains both the stop and the target, we cannot know the
   order in which they were touched. With `pessimistic_intrabar: true` we assume
   the stop was hit first. Always.
5. The trailing stop for bar i+1 is computed from bar i's close. It is never
   tightened using information from inside bar i+1.

If you change this file, keep those five rules or the results stop meaning
anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
@dataclass
class Trade:
    """One completed round trip."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    quantity: int
    lots: float
    initial_stop: float
    final_stop: float  # where the (possibly trailed) stop sat when the trade closed
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float  # net P&L as a fraction of equity at entry
    r_multiple: float  # net P&L divided by the rupee risk taken at entry
    exit_reason: str
    bars_held: int
    mae: float  # max adverse excursion, in points
    mfe: float  # max favourable excursion, in points

    @property
    def side(self) -> str:
        return "LONG" if self.direction == 1 else "SHORT"


@dataclass
class Position:
    """The single open position, if any."""

    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    initial_stop: float
    stop: float
    target: float | None
    atr_at_entry: float
    equity_at_entry: float
    risk_rupees: float
    entry_cost: float
    entry_bar: int
    breakeven_done: bool = False
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    bars: pd.DataFrame
    config: Config
    skipped: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
def _order_cost(price: float, qty: int, is_buy: bool, cfg: Config) -> float:
    """Total statutory + broker cost of ONE order, in rupees.

    Modelled on Indian index-futures charges. Slippage is NOT included here —
    it is applied to the fill price instead, which is where it actually bites.
    """
    c = cfg.costs
    turnover = price * qty

    brokerage = c.brokerage_per_order
    exchange = turnover * c.exchange_txn_pct
    sebi = turnover * c.sebi_fee_pct
    stt = turnover * c.stt_sell_pct if not is_buy else 0.0
    stamp = turnover * c.stamp_duty_pct if is_buy else 0.0
    gst = (brokerage + exchange + sebi) * c.gst_pct

    return brokerage + exchange + sebi + stt + stamp + gst


def _fill_price(price: float, is_buy: bool, cfg: Config) -> float:
    """Apply slippage. Buys fill above the quote, sells below it. Always."""
    slip = cfg.costs.slippage_pct
    return price * (1.0 + slip) if is_buy else price * (1.0 - slip)


def _position_size(
    equity: float, entry_price: float, stop_price: float, cfg: Config
) -> tuple[int, float]:
    """Risk-based sizing, capped by the leverage ceiling.

    Returns (quantity, rupee_risk). Quantity is 0 when the trade cannot be taken
    at a sane size — the caller skips the signal rather than forcing a fill.
    """
    r = cfg.risk
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or not np.isfinite(stop_distance):
        return 0, 0.0

    risk_rupees = equity * r.risk_per_trade_pct
    qty = risk_rupees / stop_distance

    # Leverage ceiling — never take a notional larger than max_leverage x equity.
    max_qty_by_leverage = (equity * r.max_leverage) / entry_price
    qty = min(qty, max_qty_by_leverage)

    if r.round_to_lots:
        lots = int(qty // r.lot_size)
        qty = lots * r.lot_size
        if lots == 0 and r.allow_min_one_lot:
            # The stop is wide enough that the risk-based size rounds to zero
            # lots. Take one lot only if that still respects the hard ceiling —
            # and only if the notional still fits inside the leverage cap.
            one_lot_risk = r.lot_size * stop_distance
            one_lot_notional = r.lot_size * entry_price
            if (
                one_lot_risk <= equity * r.max_risk_per_trade_pct
                and one_lot_notional <= equity * r.max_leverage
            ):
                qty = r.lot_size
    else:
        qty = int(qty)

    if qty <= 0:
        return 0, 0.0

    # The rupee risk we report is the risk actually taken, not the risk intended.
    return int(qty), qty * stop_distance


# --------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, cfg: Config, is_intraday: bool = True) -> BacktestResult:
    """Walk the bars one at a time and simulate the strategy.

    `df` must already have been through strategy.prepare().
    """
    r = cfg.risk
    s = cfg.strategy
    bt = cfg.backtest

    equity = float(r.starting_capital)
    position: Position | None = None
    trades: list[Trade] = []

    # Orders queued at bar i-1's close, to be filled at bar i's open.
    pending_entry: int = 0        # +1 / -1 / 0
    pending_exit_reason: str | None = None

    # Per-session state
    current_day: Any = None
    day_start_equity = equity
    day_pnl = 0.0
    trades_today = 0
    day_blocked = False

    skipped = {
        "position_open": 0,
        "max_trades_per_day": 0,
        "daily_loss_limit": 0,
        "daily_profit_target": 0,
        "size_below_one_lot": 0,
        "after_entry_cutoff": 0,
    }

    # Per-bar records for the equity curve.
    eq_time: list[pd.Timestamp] = []
    eq_value: list[float] = []
    eq_exposure: list[int] = []

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr_v = df["atr"].to_numpy(float)
    st_line = df["st_line"].to_numpy(float)
    st_dir = df["st_dir"].to_numpy(float)
    ema_f = df["ema_fast"].to_numpy(float)
    ema_s = df["ema_slow"].to_numpy(float)
    signal = df["signal"].to_numpy(int)
    can_enter = df["can_enter_time"].to_numpy(bool)
    square_off = df["is_square_off"].to_numpy(bool)
    day_key = df["session_date"].to_numpy()
    index = df.index

    n = len(df)

    def close_position(bar: int, price: float, reason: str) -> None:
        """Book the exit, update equity and the day's P&L."""
        nonlocal position, equity, day_pnl
        assert position is not None
        is_buy = position.direction == -1  # closing a short means buying back
        fill = _fill_price(price, is_buy, cfg)
        exit_cost = _order_cost(fill, position.quantity, is_buy, cfg)

        gross = (fill - position.entry_price) * position.quantity * position.direction
        total_cost = position.entry_cost + exit_cost
        net = gross - total_cost

        equity += net
        day_pnl += net

        r_mult = net / position.risk_rupees if position.risk_rupees > 0 else 0.0

        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=index[bar],
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=fill,
                quantity=position.quantity,
                lots=position.quantity / r.lot_size if r.lot_size else float("nan"),
                initial_stop=position.initial_stop,
                final_stop=position.stop,
                gross_pnl=gross,
                costs=total_cost,
                net_pnl=net,
                return_pct=net / position.equity_at_entry
                if position.equity_at_entry
                else 0.0,
                r_multiple=r_mult,
                exit_reason=reason,
                bars_held=bar - position.entry_bar,
                mae=position.mae,
                mfe=position.mfe,
            )
        )
        position = None

    def open_position(bar: int, direction: int, raw_price: float, atr_ref: float) -> bool:
        """Try to open a position. Returns True if it actually got filled.

        `atr_ref` must come from a bar that has already closed — it sizes the
        stop, and therefore the position.
        """
        nonlocal position, trades_today
        if not (np.isfinite(atr_ref) and atr_ref > 0):
            return False

        is_buy = direction == 1
        fill = _fill_price(raw_price, is_buy, cfg)
        stop = fill - direction * s.stop_atr_multiplier * atr_ref
        target = (
            fill + direction * s.target_atr_multiplier * atr_ref
            if s.target_atr_multiplier > 0
            else None
        )

        qty, risk_rupees = _position_size(equity, fill, stop, cfg)
        if qty <= 0:
            skipped["size_below_one_lot"] += 1
            return False

        position = Position(
            direction=direction,
            entry_time=index[bar],
            entry_price=fill,
            quantity=qty,
            initial_stop=stop,
            stop=stop,
            target=target,
            atr_at_entry=float(atr_ref),
            equity_at_entry=equity,
            risk_rupees=risk_rupees,
            entry_cost=_order_cost(fill, qty, is_buy, cfg),
            entry_bar=bar,
        )
        trades_today += 1
        return True

    # ----------------------------------------------------------------------
    for i in range(n):
        # ---- new session housekeeping -----------------------------------
        # Only meaningful for intraday data. On daily bars there is no "session"
        # to reset — every bar would be a new day, which would wipe the pending
        # order queue on every single bar and the system would never trade.
        if is_intraday and day_key[i] != current_day:
            current_day = day_key[i]
            day_start_equity = equity
            day_pnl = 0.0
            trades_today = 0
            day_blocked = False
            # A pending order can never survive overnight.
            pending_entry = 0
            pending_exit_reason = None

        # ---- 1. fill anything queued at the previous bar's close ---------
        if position is not None and pending_exit_reason is not None:
            close_position(i, o[i], pending_exit_reason)
            pending_exit_reason = None

        if position is None and pending_entry != 0:
            direction = pending_entry
            pending_entry = 0
            # Stop is sized off the ATR of the *signal* bar (i-1), which is the
            # last fully closed bar at the moment the decision was made.
            open_position(i, direction, o[i], atr_v[i - 1] if i > 0 else np.nan)

        # ---- 2. manage an open position through THIS bar ------------------
        if position is not None:
            d = position.direction

            # Track excursions in points for the trade report.
            if d == 1:
                position.mae = min(position.mae, l[i] - position.entry_price)
                position.mfe = max(position.mfe, h[i] - position.entry_price)
            else:
                position.mae = min(position.mae, position.entry_price - h[i])
                position.mfe = max(position.mfe, position.entry_price - l[i])

            stop_hit = (l[i] <= position.stop) if d == 1 else (h[i] >= position.stop)
            target_hit = False
            if position.target is not None:
                target_hit = (
                    (h[i] >= position.target) if d == 1 else (l[i] <= position.target)
                )

            if stop_hit and target_hit:
                # Ambiguous bar. Rule 4: assume the worse of the two.
                if bt.pessimistic_intrabar:
                    close_position(i, position.stop, "stop")
                else:
                    close_position(i, position.target, "target")
            elif stop_hit:
                close_position(i, position.stop, "stop")
            elif target_hit:
                close_position(i, position.target, "target")
            elif square_off[i]:
                close_position(i, c[i], "square_off")

        # ---- 3. at THIS bar's close, decide what to do on the NEXT bar ----
        if position is not None:
            d = position.direction

            # 3a. Break-even move — once the trade has run far enough our way,
            #     the stop goes to entry so the trade can no longer lose money
            #     (before costs).
            if not position.breakeven_done and s.breakeven_atr_multiplier > 0:
                moved = (c[i] - position.entry_price) * d
                if moved >= s.breakeven_atr_multiplier * position.atr_at_entry:
                    new_stop = position.entry_price
                    if (d == 1 and new_stop > position.stop) or (
                        d == -1 and new_stop < position.stop
                    ):
                        position.stop = new_stop
                    position.breakeven_done = True

            # 3b. Supertrend trail, in "intrabar" mode only: the Supertrend line
            #     becomes the resting stop order. The stop only ever moves in our
            #     favour — a trailing stop that can loosen is not a stop.
            #     In "close" mode the line is not a stop at all; the exit fires
            #     from the flip in 3c instead. See StrategyConfig for why that
            #     distinction is the single most important switch in the config.
            if (
                s.use_supertrend_trail
                and s.supertrend_trail_mode == "intrabar"
                and np.isfinite(st_line[i])
            ):
                if d == 1 and st_dir[i] == 1 and st_line[i] > position.stop:
                    position.stop = float(st_line[i])
                elif d == -1 and st_dir[i] == -1 and st_line[i] < position.stop:
                    position.stop = float(st_line[i])

            # 3c. Signal-based exits, filled at the next bar's open.
            if st_dir[i] == -d:
                pending_exit_reason = "supertrend_flip"
            elif s.exit_on_ema_cross and (
                (d == 1 and ema_f[i] < ema_s[i]) or (d == -1 and ema_f[i] > ema_s[i])
            ):
                pending_exit_reason = "ema_cross"

        # ---- 4. entry decision for the next bar ---------------------------
        # Only when flat, not already exiting, and the session still allows it.
        if position is None and pending_exit_reason is None and signal[i] != 0:
            if is_intraday and day_blocked:
                skipped["daily_loss_limit"] += 1
            elif is_intraday and trades_today >= r.max_trades_per_day:
                skipped["max_trades_per_day"] += 1
            elif not bt.next_bar_execution:
                # Same-bar-close fills. Kept only so the Python run can be
                # compared like-for-like with a TradingView strategy tester that
                # fills on close. It is optimistic — you cannot trade a close you
                # have only just observed — so leave next_bar_execution on for
                # any result you intend to believe.
                if not (is_intraday and square_off[i]):
                    open_position(i, int(signal[i]), c[i], atr_v[i])
            elif i + 1 >= n or (is_intraday and day_key[i + 1] != day_key[i]):
                # No next bar in this session to fill against.
                skipped["after_entry_cutoff"] += 1
            elif is_intraday and square_off[i + 1]:
                # Filling into the square-off bar would open and close instantly.
                skipped["after_entry_cutoff"] += 1
            else:
                pending_entry = int(signal[i])

        # ---- 5. daily circuit breakers (intraday only) ---------------------
        if not is_intraday:
            pass
        elif r.daily_loss_limit_pct > 0 and day_pnl <= -r.daily_loss_limit_pct * day_start_equity:
            if not day_blocked:
                log.debug("Daily loss limit hit on %s (P&L %.0f)", current_day, day_pnl)
            day_blocked = True
            pending_entry = 0
        if (
            r.daily_profit_target_pct > 0
            and day_pnl >= r.daily_profit_target_pct * day_start_equity
        ):
            day_blocked = True
            pending_entry = 0

        # ---- 6. mark to market --------------------------------------------
        if position is not None:
            unreal = (c[i] - position.entry_price) * position.quantity * position.direction
            mtm = equity + unreal - position.entry_cost
        else:
            mtm = equity
        eq_time.append(index[i])
        eq_value.append(mtm)
        eq_exposure.append(0 if position is None else position.direction)

    # Anything still open at the very end of the data is closed at the last close.
    if position is not None:
        close_position(n - 1, c[n - 1], "end_of_data")
        eq_value[-1] = equity

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    if not trades_df.empty:
        trades_df["side"] = np.where(trades_df["direction"] == 1, "LONG", "SHORT")
        trades_df["cum_pnl"] = trades_df["net_pnl"].cumsum()
        trades_df["equity_after"] = r.starting_capital + trades_df["cum_pnl"]

    equity_df = pd.DataFrame(
        {"equity": eq_value, "exposure": eq_exposure}, index=pd.DatetimeIndex(eq_time)
    )
    equity_df.index.name = "datetime"

    log.info(
        "Backtest complete: %d trades, final equity %.0f (start %.0f)",
        len(trades_df), equity, r.starting_capital,
    )
    for k, v in skipped.items():
        if v:
            log.info("  signals skipped (%s): %d", k, v)

    return BacktestResult(
        trades=trades_df,
        equity_curve=equity_df,
        bars=df,
        config=cfg,
        skipped=skipped,
    )
