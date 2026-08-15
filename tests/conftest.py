"""Shared fixtures. Makes `src/` importable without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def make_session_index(days: int = 20, freq_minutes: int = 15) -> pd.DatetimeIndex:
    """A realistic NSE intraday index: 09:15 to 15:30 IST, weekdays only."""
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2025-01-01", tz="Asia/Kolkata")
    made = 0
    while made < days:
        if day.weekday() < 5:
            t = day + pd.Timedelta(hours=9, minutes=15)
            end = day + pd.Timedelta(hours=15, minutes=30)
            while t <= end:
                stamps.append(t)
                t += pd.Timedelta(minutes=freq_minutes)
            made += 1
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps, name="datetime")


def synth_ohlcv(index: pd.DatetimeIndex, seed: int = 7, start: float = 20000.0,
                drift: float = 0.00004) -> pd.DataFrame:
    """Deterministic random-walk OHLCV. Same seed always gives the same bars."""
    rng = np.random.default_rng(seed)
    n = len(index)
    steps = rng.normal(drift, 0.0012, n)
    close = start * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0.0009, 0.0004, n)) * close
    open_ = np.concatenate([[close[0]], close[:-1]]) * (
        1 + rng.normal(0, 0.0002, n)
    )
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": rng.integers(1000, 50000, n).astype(float),
        },
        index=index,
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return synth_ohlcv(make_session_index(30))


@pytest.fixture
def cfg():
    from nifty_algo import Config

    c = Config()
    c.data.interval = "15m"
    c.risk.starting_capital = 1_000_000.0
    return c
