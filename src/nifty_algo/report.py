"""
Backtest report generation.

Produces, into `output/`:
  <tag>_report.html   a single self-contained page (charts embedded as base64,
                      no internet needed to open it)
  <tag>_trades.csv    every trade, so you can audit any number in the report
  <tag>_equity.csv    the bar-by-bar equity curve
  <tag>_metrics.json  the metric block, machine readable

The HTML is deliberately dependency-free — open it in any browser, email it,
archive it. Nothing loads from a CDN.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestResult  # noqa: E402
from . import metrics as mx  # noqa: E402

log = logging.getLogger(__name__)

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
}

GREEN = "#0f9960"
RED = "#d9534f"
BLUE = "#2b6cb0"
GREY = "#8a94a6"


def _fig_to_b64(fig) -> str:
    """Render a figure to a base64 PNG string and close it.

    `pad_inches` is generous on purpose: with a tight bbox, rotated y-axis
    labels on stacked subplots get shaved off at the left edge.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
def plot_equity_and_drawdown(equity: pd.DataFrame, starting_capital: float) -> str:
    eq = equity["equity"].astype(float)
    dd = mx.drawdown_series(eq) * 100.0

    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True, layout="constrained",
            gridspec_kw={"height_ratios": [2.2, 1]},
        )

        ax1.plot(eq.index, eq.to_numpy(), color=BLUE, linewidth=1.3)
        ax1.axhline(starting_capital, color=GREY, linewidth=0.9, linestyle="--")
        ax1.fill_between(
            eq.index, starting_capital, eq.to_numpy(),
            where=(eq.to_numpy() >= starting_capital),
            color=GREEN, alpha=0.12, interpolate=True,
        )
        ax1.fill_between(
            eq.index, starting_capital, eq.to_numpy(),
            where=(eq.to_numpy() < starting_capital),
            color=RED, alpha=0.12, interpolate=True,
        )
        ax1.set_ylabel("Equity (INR)")
        ax1.set_title("Equity curve", loc="left", fontweight="bold")

        ax2.fill_between(dd.index, dd.to_numpy(), 0.0, color=RED, alpha=0.35)
        ax2.plot(dd.index, dd.to_numpy(), color=RED, linewidth=0.9)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown from peak", loc="left", fontweight="bold")

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
        # Explicit rotation rather than fig.autofmt_xdate(), which calls
        # subplots_adjust and fights with the constrained layout engine.
        plt.setp(ax2.get_xticklabels(), rotation=35, ha="right")
        return _fig_to_b64(fig)


def plot_trade_distribution(trades: pd.DataFrame) -> str:
    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")

        pnl = trades["net_pnl"].astype(float)
        bins = min(40, max(8, len(pnl) // 2))
        ax1.hist(pnl[pnl > 0], bins=bins, color=GREEN, alpha=0.75, label="Wins")
        ax1.hist(pnl[pnl <= 0], bins=bins, color=RED, alpha=0.75, label="Losses")
        ax1.axvline(0, color=GREY, linewidth=0.9)
        ax1.set_xlabel("Net P&L per trade (INR)")
        ax1.set_ylabel("Count")
        ax1.set_title("Trade P&L distribution", loc="left", fontweight="bold")
        ax1.legend(frameon=False, fontsize=8)

        r = trades["r_multiple"].astype(float)
        colors = [GREEN if v > 0 else RED for v in r]
        ax2.bar(range(len(r)), r.to_numpy(), color=colors, width=0.9)
        ax2.axhline(0, color=GREY, linewidth=0.9)
        ax2.axhline(-1, color=GREY, linewidth=0.7, linestyle="--")
        ax2.set_xlabel("Trade number")
        ax2.set_ylabel("R multiple")
        ax2.set_title("Outcome in R (P&L / risk taken)", loc="left", fontweight="bold")

        return _fig_to_b64(fig)


def plot_monthly(monthly: pd.DataFrame) -> str:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 3.0), layout="constrained")
        vals = monthly["return_pct"].to_numpy(float)
        colors = [GREEN if v > 0 else RED for v in vals]
        ax.bar(monthly.index.astype(str), vals, color=colors)
        ax.axhline(0, color=GREY, linewidth=0.9)
        ax.set_ylabel("Return (%)")
        ax.set_title("Monthly return", loc="left", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        return _fig_to_b64(fig)


def plot_price_with_trades(
    bars: pd.DataFrame, trades: pd.DataFrame, max_bars: int = 900
) -> str:
    """Last slice of price with entry/exit markers — a visual sanity check.

    If the arrows are not sitting on breakouts in the direction of the trend,
    the code is not doing what the documentation says it does.
    """
    view = bars.iloc[-max_bars:]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
        x = np.arange(len(view))  # positional x removes the weekend/overnight gaps
        ax.plot(x, view["close"].to_numpy(), color="#333", linewidth=0.9, label="Close")
        if "st_line" in view:
            up = view["st_dir"] == 1
            ax.plot(x, np.where(up, view["st_line"], np.nan), color=GREEN,
                    linewidth=0.9, label="Supertrend (up)")
            ax.plot(x, np.where(~up, view["st_line"], np.nan), color=RED,
                    linewidth=0.9, label="Supertrend (down)")

        pos = {ts: i for i, ts in enumerate(view.index)}
        for _, t in trades.iterrows():
            xi = pos.get(t["entry_time"])
            xo = pos.get(t["exit_time"])
            if xi is None:
                continue
            marker = "^" if t["direction"] == 1 else "v"
            colour = GREEN if t["direction"] == 1 else RED
            ax.scatter([xi], [t["entry_price"]], marker=marker, s=70,
                       color=colour, zorder=5, edgecolors="white", linewidths=0.6)
            if xo is not None:
                ax.scatter([xo], [t["exit_price"]], marker="x", s=48,
                           color="#111", zorder=5, linewidths=1.2)
                ax.plot([xi, xo], [t["entry_price"], t["exit_price"]],
                        color=colour, linewidth=0.8, alpha=0.5)

        # Date labels on a positional axis.
        step = max(1, len(view) // 10)
        ticks = list(range(0, len(view), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [view.index[i].strftime("%d %b %H:%M") for i in ticks],
            rotation=45, ha="right",
        )
        ax.set_ylabel("Price")
        ax.set_title(
            f"Signals on the last {len(view)} bars "
            "(triangle = entry, cross = exit)",
            loc="left", fontweight="bold",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        return _fig_to_b64(fig)


# --------------------------------------------------------------------------
def _fmt(v, kind: str = "num") -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "&ndash;"
    if kind == "money":
        return f"{v:,.0f}"
    if kind == "pct":
        return f"{v:,.2f}%"
    if kind == "ratio":
        return f"{v:,.2f}"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:,.2f}"


def _metric_card(label: str, value: str, good: bool | None = None) -> str:
    cls = "" if good is None else (" good" if good else " bad")
    return (
        f'<div class="card{cls}"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{value}</div></div>'
    )


def _table(df: pd.DataFrame, float_fmt: str = "{:,.2f}") -> str:
    if df is None or df.empty:
        return "<p class='muted'>No data.</p>"
    return df.to_html(
        classes="data", border=0, float_format=lambda v: float_fmt.format(v)
    )


CSS = """
:root { --fg:#1a1f2b; --muted:#6b7280; --line:#e5e7eb; --good:#0f9960; --bad:#d9534f; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--fg); margin: 0; padding: 32px; background: #f7f8fa; line-height: 1.5; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 34px 0 12px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h3 { font-size: 14px; margin: 22px 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.sub { color: var(--muted); margin: 0 0 8px; font-size: 14px; }
.panel { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 22px 26px; margin-bottom: 18px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.card { background: #fff; border: 1px solid var(--line); border-left: 3px solid var(--muted);
        border-radius: 7px; padding: 11px 13px; }
.card.good { border-left-color: var(--good); }
.card.bad  { border-left-color: var(--bad); }
.card-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.card-value { font-size: 19px; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; }
img { max-width: 100%; height: auto; display: block; margin: 10px 0 4px; }
table.data { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
table.data th, table.data td { padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: right; }
table.data th { background: #fafbfc; font-weight: 600; color: var(--muted);
                text-transform: uppercase; font-size: 11px; letter-spacing: .03em; }
table.data td:first-child, table.data th:first-child { text-align: left; }
table.data tbody tr:hover { background: #fafbfc; }
.muted { color: var(--muted); font-size: 13px; }
.note { background: #fffbea; border: 1px solid #f5e2a3; border-radius: 7px;
        padding: 12px 16px; font-size: 13px; margin: 14px 0; }
pre { background: #f4f5f7; border: 1px solid var(--line); border-radius: 7px;
      padding: 14px; overflow-x: auto; font-size: 12px; line-height: 1.45; }
footer { color: var(--muted); font-size: 12px; margin-top: 30px; text-align: center; }
"""


def build_html(result: BacktestResult, title: str, tag: str) -> str:
    cfg = result.config
    trades = result.trades
    equity = result.equity_curve
    m = mx.compute_metrics(trades, equity, cfg.risk.starting_capital)

    charts = [
        ("Equity and drawdown", plot_equity_and_drawdown(equity, cfg.risk.starting_capital)),
    ]
    if not trades.empty:
        charts.append(("Trade distribution", plot_trade_distribution(trades)))
    monthly = mx.monthly_returns(equity)
    if len(monthly) > 1:
        charts.append(("Monthly returns", plot_monthly(monthly)))
    charts.append(("Signals on the chart", plot_price_with_trades(result.bars, trades)))

    cards = "".join(
        [
            _metric_card("Net profit", "&#8377; " + _fmt(m["net_profit"], "money"),
                         m["net_profit"] > 0),
            _metric_card("Total return", _fmt(m["total_return_pct"], "pct"),
                         m["total_return_pct"] > 0),
            _metric_card("CAGR", _fmt(m["cagr_pct"], "pct"), m["cagr_pct"] > 0),
            _metric_card("Max drawdown", _fmt(m["max_drawdown_pct"], "pct"),
                         m["max_drawdown_pct"] > -15),
            _metric_card("Sharpe", _fmt(m["sharpe"], "ratio"),
                         (m["sharpe"] or 0) > 1),
            _metric_card("Sortino", _fmt(m["sortino"], "ratio"),
                         (m["sortino"] or 0) > 1),
            _metric_card("Calmar", _fmt(m["calmar"], "ratio"), None),
            _metric_card("Profit factor", _fmt(m["profit_factor"], "ratio"),
                         m["profit_factor"] > 1),
            _metric_card("Win rate", _fmt(m["win_rate_pct"], "pct"), None),
            _metric_card("Expectancy / trade", "&#8377; " + _fmt(m["expectancy"], "money"),
                         m["expectancy"] > 0),
            _metric_card("Expectancy (R)", _fmt(m["expectancy_r"], "ratio"),
                         m["expectancy_r"] > 0),
            _metric_card("Total trades", _fmt(m["total_trades"], "int"), None),
        ]
    )

    detail_rows = [
        ("Period", f"{m['start_date'][:16]} &rarr; {m['end_date'][:16]}"),
        ("Trading sessions", _fmt(m["trading_sessions"], "int")),
        ("Trades per session", _fmt(m.get("trades_per_session", 0), "ratio")),
        ("Winning / losing trades",
         f"{m.get('winning_trades', 0)} / {m.get('losing_trades', 0)}"),
        ("Average win", "&#8377; " + _fmt(m["avg_win"], "money")),
        ("Average loss", "&#8377; " + _fmt(m["avg_loss"], "money")),
        ("Payoff ratio (avg win / avg loss)", _fmt(m["payoff_ratio"], "ratio")),
        ("Largest win", "&#8377; " + _fmt(m["largest_win"], "money")),
        ("Largest loss", "&#8377; " + _fmt(m["largest_loss"], "money")),
        ("Max consecutive wins", _fmt(m["max_consecutive_wins"], "int")),
        ("Max consecutive losses", _fmt(m["max_consecutive_losses"], "int")),
        ("Average bars held", _fmt(m["avg_bars_held"], "ratio")),
        ("Annualised volatility", _fmt(m.get("annual_volatility_pct"), "pct")),
        ("Longest drawdown (bars)", _fmt(m["max_drawdown_bars"], "int")),
        ("Gross profit (before costs)", "&#8377; " + _fmt(m["gross_profit"], "money")),
        ("Total transaction costs", "&#8377; " + _fmt(m["total_costs"], "money")),
        ("Costs as % of gross", _fmt(m.get("cost_pct_of_gross"), "pct")),
        ("Long trades / net P&L",
         f"{m['long_trades']} &nbsp;/&nbsp; &#8377; {_fmt(m.get('long_net_pnl'), 'money')}"),
        ("Short trades / net P&L",
         f"{m['short_trades']} &nbsp;/&nbsp; &#8377; {_fmt(m.get('short_net_pnl'), 'money')}"),
        ("Long / short win rate",
         f"{_fmt(m['long_win_rate_pct'], 'pct')} &nbsp;/&nbsp; {_fmt(m['short_win_rate_pct'], 'pct')}"),
    ]
    detail_html = "<table class='data'><tbody>" + "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in detail_rows
    ) + "</tbody></table>"

    chart_html = "".join(
        f"<h3>{html.escape(name)}</h3><img alt='{html.escape(name)}' "
        f"src='data:image/png;base64,{b64}'/>"
        for name, b64 in charts
    )

    exits = mx.exit_reason_breakdown(trades)
    exits_html = _table(exits)

    monthly_html = _table(monthly[["net_pnl", "return_pct"]]) if len(monthly) else ""

    # Recent trades — the full list is in the CSV.
    if not trades.empty:
        show = trades.tail(25).copy()
        show["entry_time"] = pd.to_datetime(show["entry_time"]).dt.strftime("%d-%b %H:%M")
        show["exit_time"] = pd.to_datetime(show["exit_time"]).dt.strftime("%d-%b %H:%M")
        show = show[[
            "entry_time", "exit_time", "side", "quantity", "entry_price",
            "exit_price", "initial_stop", "net_pnl", "r_multiple", "exit_reason",
        ]]
        trades_html = _table(show)
    else:
        trades_html = "<p class='muted'>No trades were taken.</p>"

    skipped_rows = "".join(
        f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td>{v}</td></tr>"
        for k, v in result.skipped.items() if v
    )
    skipped_html = (
        f"<table class='data'><thead><tr><th>Reason</th><th>Signals</th></tr></thead>"
        f"<tbody>{skipped_rows}</tbody></table>"
        if skipped_rows
        else "<p class='muted'>Every valid signal was acted on.</p>"
    )

    cfg_json = html.escape(json.dumps(cfg.to_dict(), indent=2, default=str))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><div class="wrap">

<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(cfg.data.name)} &middot; {html.escape(cfg.data.interval)} bars
&middot; intraday trend-following &middot; all figures are net of modelled costs and slippage</p>

<div class="panel"><div class="cards">{cards}</div></div>

<h2>Charts</h2>
<div class="panel">{chart_html}</div>

<h2>Detailed statistics</h2>
<div class="panel">{detail_html}</div>

<h2>How trades ended</h2>
<div class="panel">
<p class="muted">A healthy trend system exits mostly on the trailing stop and the
Supertrend flip. A pile-up on <code>square_off</code> means trades are not
running; a pile-up on <code>stop</code> means the entry filter is letting
low-quality breakouts through.</p>
{exits_html}
</div>

<h2>Signals that were not traded</h2>
<div class="panel">
<p class="muted">Risk management refusing a signal is the system working, not
failing. This is where the daily loss limit and the trade cap show up.</p>
{skipped_html}
</div>

<h2>Monthly breakdown</h2>
<div class="panel">{monthly_html}</div>

<h2>Last 25 trades</h2>
<div class="panel">
<p class="muted">Complete list in <code>{html.escape(tag)}_trades.csv</code>.</p>
{trades_html}
</div>

<h2>Exact configuration used</h2>
<div class="panel">
<p class="muted">Reproduce this report by running the backtest with this config.
Every number above follows from it.</p>
<pre>{cfg_json}</pre>
</div>

<div class="note"><strong>Read this before risking money.</strong> A backtest is a
simulation, not a promise. Results depend on the data vendor, the assumed slippage
and the cost model above &mdash; change any of them and the numbers change. Index
values are not directly tradeable: real fills happen in futures or options, which
have their own spread, roll and (for options) time decay. Forward-test on paper
before going live.</div>

<footer>Generated by nifty-trend-algo &middot; {html.escape(tag)}</footer>
</div></body></html>"""


def write_report(result: BacktestResult, tag: str, title: str) -> dict[str, Path]:
    """Write the HTML report plus the CSV/JSON side-cars. Returns the paths."""
    out_dir = Path(result.config.backtest.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    html_path = out_dir / f"{tag}_report.html"
    html_path.write_text(build_html(result, title, tag), encoding="utf-8")
    paths["html"] = html_path

    trades_path = out_dir / f"{tag}_trades.csv"
    result.trades.to_csv(trades_path, index=False)
    paths["trades"] = trades_path

    equity_path = out_dir / f"{tag}_equity.csv"
    result.equity_curve.to_csv(equity_path)
    paths["equity"] = equity_path

    metrics_path = out_dir / f"{tag}_metrics.json"
    m = mx.compute_metrics(
        result.trades, result.equity_curve, result.config.risk.starting_capital
    )
    metrics_path.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    paths["metrics"] = metrics_path

    log.info("Report written to %s", html_path)
    return paths
