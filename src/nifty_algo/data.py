"""
Data loading.

Two sources, one output shape:

  * Yahoo Finance (yfinance) — free, zero setup, but only ~60 days of history
    for intraday bars of 30 minutes or less, and ~730 days for hourly.
  * A CSV you supply — your broker's historical dump, a purchased dataset,
    anything. This is the route to a multi-year intraday backtest.

Whichever you use, `load_ohlcv()` returns the exact same DataFrame:

    index : tz-aware DatetimeIndex in Asia/Kolkata, sorted, unique
    cols  : open, high, low, close, volume  (float64, lowercase)

Downloads are cached to Parquet/CSV on disk. That is what makes the backtest
reproducible: the second run reads the cached bars instead of re-hitting Yahoo,
so you get identical numbers even though Yahoo's window slides forward daily.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from .config import DataConfig, SessionConfig

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------
def _cache_key(cfg: DataConfig) -> str:
    """Stable filename for a given download request."""
    parts = [
        cfg.symbol,
        cfg.interval,
        cfg.period or "",
        cfg.start or "",
        cfg.end or "",
    ]
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    safe_symbol = cfg.symbol.replace("^", "").replace("/", "_")
    return f"{safe_symbol}_{cfg.interval}_{digest}.csv"


def _normalise(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Coerce any reasonable OHLCV frame into the canonical shape."""
    df = df.copy()

    # yfinance returns a MultiIndex column frame when given a list of tickers,
    # and sometimes even for a single ticker depending on version. Flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Accept 'adj_close' but never use it — index data is not adjusted anyway.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Price data is missing required column(s): {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    df = df[REQUIRED_COLUMNS]

    # Timezone: Yahoo hands back UTC for intraday and naive dates for daily.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        # Naive timestamps are assumed to already be exchange local time.
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    df.index.name = "datetime"

    df = df.astype("float64")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Drop bars with no price. Yahoo pads holidays and pre-open with NaN rows.
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0]

    # Sanity: high must bound the bar. A handful of vendor ticks violate this;
    # repair rather than discard so the session stays contiguous.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    return df


# --------------------------------------------------------------------------
def load_from_csv(path: str | Path, tz: str) -> pd.DataFrame:
    """Load bars from a CSV.

    Expected header (case-insensitive, extra columns ignored):
        datetime,open,high,low,close,volume

    `datetime` may be tz-aware ISO-8601 or plain 'YYYY-MM-DD HH:MM:SS'. Plain
    timestamps are assumed to be exchange local time (IST) — which is what every
    Indian broker exports.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Find the timestamp column under any of its common names.
    ts_col = next(
        (c for c in ("datetime", "date", "timestamp", "time") if c in df.columns),
        None,
    )
    if ts_col is None:
        raise ValueError(
            "CSV needs a timestamp column named one of: datetime, date, timestamp, time"
        )

    df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", utc=False)
    df = df.set_index(ts_col)
    return _normalise(df, tz)


def load_from_yahoo(cfg: DataConfig) -> pd.DataFrame:
    """Download bars from Yahoo Finance, with an on-disk cache."""
    import yfinance as yf  # imported lazily so CSV-only users need no network

    cache_file = Path(cfg.cache_dir) / _cache_key(cfg)
    if cfg.use_cache and cache_file.exists():
        log.info("Using cached data: %s", cache_file)
        return load_from_csv(cache_file, cfg.timezone)

    log.info(
        "Downloading %s %s bars from Yahoo Finance ...", cfg.symbol, cfg.interval
    )
    kwargs = dict(
        tickers=cfg.symbol,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if cfg.start:
        kwargs["start"] = cfg.start
        if cfg.end:
            kwargs["end"] = cfg.end
    else:
        kwargs["period"] = cfg.period

    raw = yf.download(**kwargs)
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"Yahoo Finance returned no data for {cfg.symbol} at {cfg.interval}. "
            "Intraday history is capped at 60 days for intervals of 30m or less "
            "and 730 days for 1h — check `period` in your config."
        )

    df = _normalise(raw, cfg.timezone)

    if cfg.use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out.index = out.index.strftime("%Y-%m-%d %H:%M:%S%z")
        out.to_csv(cache_file, index_label="datetime")
        log.info("Cached %d bars to %s", len(df), cache_file)

    return df


def load_ohlcv(cfg: DataConfig) -> pd.DataFrame:
    """Entry point: CSV if one is configured, otherwise Yahoo Finance."""
    if cfg.csv_path:
        log.info("Loading data from CSV: %s", cfg.csv_path)
        return load_from_csv(cfg.csv_path, cfg.timezone)
    return load_from_yahoo(cfg)


# --------------------------------------------------------------------------
def filter_session(
    df: pd.DataFrame, session: SessionConfig, is_intraday: bool
) -> pd.DataFrame:
    """Keep only bars inside the regular trading session.

    Yahoo occasionally returns a stray bar stamped outside 09:15-15:30 IST.
    Those bars would produce trades you could never actually take.
    """
    if not is_intraday:
        return df

    open_t = pd.Timestamp(session.market_open).time()
    close_t = pd.Timestamp(session.market_close).time()
    mask = (df.index.time >= open_t) & (df.index.time <= close_t)
    dropped = int((~mask).sum())
    if dropped:
        log.info("Dropped %d bar(s) outside the %s-%s session",
                 dropped, session.market_open, session.market_close)
    return df[mask]


def is_intraday_interval(interval: str) -> bool:
    """True for minute/hour bars, False for daily and above."""
    return interval.endswith("m") or interval.endswith("h")


def resample_htf(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """Aggregate the signal timeframe up to the higher timeframe.

    Uses a *rolling* aggregation rather than pandas' calendar resample, because
    the NSE session (09:15-15:30) does not divide cleanly into clock hours and a
    calendar resample would silently mix bars across the lunch-less Indian
    session boundary. Rolling windows keep every higher-timeframe value aligned
    to the signal bar that closed it — and, critically, causal.
    """
    if multiplier <= 1:
        return df.copy()

    out = pd.DataFrame(index=df.index)
    out["open"] = df["open"].shift(multiplier - 1)
    out["high"] = df["high"].rolling(multiplier, min_periods=multiplier).max()
    out["low"] = df["low"].rolling(multiplier, min_periods=multiplier).min()
    out["close"] = df["close"]
    out["volume"] = df["volume"].rolling(multiplier, min_periods=multiplier).sum()
    return out
