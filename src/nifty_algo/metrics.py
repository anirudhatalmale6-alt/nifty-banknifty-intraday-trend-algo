"""
Performance metrics.

Every number the report shows is computed here, from the trade list and the
bar-by-bar equity curve produced by backtest.py. Nothing is annualised by a
hand-waved constant — the periods-per-year figure is derived from the actual
timestamps in the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer how many bars there are in a trading year from the data itself."""
    if len(index) < 3:
        return float(TRADING_DAYS_PER_YEAR)
    days = pd.Series(index.date).nunique()
    if days <= 0:
        return float(TRADING_DAYS_PER_YEAR)
    bars_per_day = len(index) / days
    return bars_per_day * TRADING_DAYS_PER_YEAR


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak, at every bar."""
    peak = equity.cummax()
    return equity / peak - 1.0


def compute_metrics(
    trades: pd.DataFrame, equity: pd.DataFrame, starting_capital: float
) -> dict:
    """Return the full metric set as a flat dict of plain Python numbers."""
    eq = equity["equity"].astype(float)
    m: dict = {}

    # --- Headline ---------------------------------------------------------
    final_equity = float(eq.iloc[-1]) if len(eq) else starting_capital
    net_profit = final_equity - starting_capital
    m["starting_capital"] = starting_capital
    m["final_equity"] = final_equity
    m["net_profit"] = net_profit
    m["total_return_pct"] = net_profit / starting_capital * 100.0

    # --- Time span --------------------------------------------------------
    if len(eq):
        span_days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400.0, 1.0)
        years = span_days / 365.25
        sessions = pd.Series(eq.index.date).nunique()
    else:
        span_days, years, sessions = 0.0, 0.0, 0
    m["start_date"] = str(eq.index[0]) if len(eq) else ""
    m["end_date"] = str(eq.index[-1]) if len(eq) else ""
    m["calendar_days"] = span_days
    m["trading_sessions"] = int(sessions)
    m["years"] = years

    m["cagr_pct"] = (
        ((final_equity / starting_capital) ** (1.0 / years) - 1.0) * 100.0
        if years > 0 and final_equity > 0
        else 0.0
    )

    # --- Risk -------------------------------------------------------------
    dd = drawdown_series(eq)
    m["max_drawdown_pct"] = float(dd.min() * 100.0) if len(dd) else 0.0
    peak = eq.cummax()
    m["max_drawdown_value"] = float((eq - peak).min()) if len(eq) else 0.0

    # Longest stretch (in bars) spent below a previous equity peak.
    under = dd < -1e-12
    if under.any():
        grp = (~under).cumsum()
        m["max_drawdown_bars"] = int(under.groupby(grp).sum().max())
    else:
        m["max_drawdown_bars"] = 0

    rets = eq.pct_change().dropna()
    ppy = _periods_per_year(eq.index)
    if len(rets) > 1 and rets.std(ddof=1) > 0:
        m["sharpe"] = float(rets.mean() / rets.std(ddof=1) * np.sqrt(ppy))
        downside = rets[rets < 0]
        m["sortino"] = (
            float(rets.mean() / downside.std(ddof=1) * np.sqrt(ppy))
            if len(downside) > 1 and downside.std(ddof=1) > 0
            else float("nan")
        )
        m["annual_volatility_pct"] = float(rets.std(ddof=1) * np.sqrt(ppy) * 100.0)
    else:
        m["sharpe"] = float("nan")
        m["sortino"] = float("nan")
        m["annual_volatility_pct"] = float("nan")

    m["calmar"] = (
        m["cagr_pct"] / abs(m["max_drawdown_pct"])
        if m["max_drawdown_pct"] < 0
        else float("nan")
    )

    # --- Trade statistics --------------------------------------------------
    n = len(trades)
    m["total_trades"] = n
    if n == 0:
        m.update(
            {
                "win_rate_pct": 0.0, "profit_factor": float("nan"),
                "expectancy": 0.0, "expectancy_r": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "payoff_ratio": float("nan"),
                "largest_win": 0.0, "largest_loss": 0.0,
                "max_consecutive_wins": 0, "max_consecutive_losses": 0,
                "avg_bars_held": 0.0, "total_costs": 0.0,
                "gross_profit": 0.0, "long_trades": 0, "short_trades": 0,
                "long_win_rate_pct": 0.0, "short_win_rate_pct": 0.0,
                "trades_per_session": 0.0,
            }
        )
        return m

    pnl = trades["net_pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    m["winning_trades"] = int(len(wins))
    m["losing_trades"] = int(len(losses))
    m["win_rate_pct"] = len(wins) / n * 100.0
    m["gross_profit"] = float(trades["gross_pnl"].sum())
    m["total_costs"] = float(trades["costs"].sum())
    m["cost_pct_of_gross"] = (
        m["total_costs"] / abs(m["gross_profit"]) * 100.0
        if m["gross_profit"] != 0
        else float("nan")
    )

    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    m["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else float("inf")
    m["expectancy"] = float(pnl.mean())
    m["expectancy_r"] = float(trades["r_multiple"].mean())
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    m["payoff_ratio"] = (
        abs(m["avg_win"] / m["avg_loss"]) if m["avg_loss"] != 0 else float("nan")
    )
    m["largest_win"] = float(pnl.max())
    m["largest_loss"] = float(pnl.min())
    m["avg_bars_held"] = float(trades["bars_held"].mean())
    m["avg_r_win"] = float(trades.loc[pnl > 0, "r_multiple"].mean()) if len(wins) else 0.0
    m["avg_r_loss"] = float(trades.loc[pnl <= 0, "r_multiple"].mean()) if len(losses) else 0.0

    # Consecutive runs.
    is_win = (pnl > 0).to_numpy()
    m["max_consecutive_wins"] = _max_run(is_win, True)
    m["max_consecutive_losses"] = _max_run(is_win, False)

    # Long vs short split — a trend system should not be lopsided by accident.
    longs = trades[trades["direction"] == 1]
    shorts = trades[trades["direction"] == -1]
    m["long_trades"] = int(len(longs))
    m["short_trades"] = int(len(shorts))
    m["long_net_pnl"] = float(longs["net_pnl"].sum()) if len(longs) else 0.0
    m["short_net_pnl"] = float(shorts["net_pnl"].sum()) if len(shorts) else 0.0
    m["long_win_rate_pct"] = (
        float((longs["net_pnl"] > 0).mean() * 100.0) if len(longs) else 0.0
    )
    m["short_win_rate_pct"] = (
        float((shorts["net_pnl"] > 0).mean() * 100.0) if len(shorts) else 0.0
    )
    m["trades_per_session"] = n / sessions if sessions else 0.0

    return m


def _max_run(flags: np.ndarray, value: bool) -> int:
    """Longest consecutive run of `value` in a boolean array."""
    best = run = 0
    for f in flags:
        if bool(f) == value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """How each trade ended, and what that exit was worth on average.

    This is the table that tells you whether the strategy is actually working
    the way it is documented — if 'square_off' dominates, the trades are not
    trending; if 'stop' dominates, the entry filter is too loose.
    """
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("exit_reason")["net_pnl"]
    out = pd.DataFrame(
        {
            "trades": g.size(),
            "net_pnl": g.sum(),
            "avg_pnl": g.mean(),
            "win_rate_pct": trades.groupby("exit_reason")["net_pnl"].apply(
                lambda x: (x > 0).mean() * 100.0
            ),
        }
    )
    return out.sort_values("trades", ascending=False)


def monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month P&L in rupees and percent, from the equity curve."""
    eq = equity["equity"].astype(float)
    if eq.empty:
        return pd.DataFrame()
    monthly = eq.resample("ME").last()
    first = pd.Series([eq.iloc[0]], index=[monthly.index[0] - pd.offsets.MonthEnd(1)])
    joined = pd.concat([first, monthly])
    pnl = joined.diff().dropna()
    pct = joined.pct_change().dropna() * 100.0
    return pd.DataFrame(
        {"net_pnl": pnl, "return_pct": pct, "equity": monthly}
    ).set_index(monthly.index.strftime("%Y-%m"))


def daily_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Realised P&L per trading session, plus how many trades it took."""
    if trades.empty:
        return pd.DataFrame()
    d = trades.copy()
    d["date"] = pd.to_datetime(d["exit_time"]).dt.date
    g = d.groupby("date")["net_pnl"]
    return pd.DataFrame({"net_pnl": g.sum(), "trades": g.size()})
