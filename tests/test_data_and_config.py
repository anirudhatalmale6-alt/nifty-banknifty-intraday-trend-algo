"""
Data loading and configuration tests.

Boring, and exactly the layer where a silent failure does the most damage: a
mis-parsed timestamp column shifts every bar by 5½ hours and the backtest still
runs, still produces a curve, and is completely meaningless.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from nifty_algo import Config, filter_session, is_intraday_interval
from nifty_algo.config import StrategyConfig
from nifty_algo.data import load_from_csv, resample_htf
from conftest import make_session_index, synth_ohlcv


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def test_shipped_configs_load(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ("nifty.yaml", "banknifty.yaml"):
        cfg = Config.from_yaml(root / "config" / name)
        assert cfg.data.symbol.startswith("^")
        assert cfg.risk.lot_size > 0
        assert cfg.strategy.stop_atr_multiplier > 0


def test_unknown_config_key_is_rejected(tmp_path):
    """A typo must fail loudly, not be silently ignored."""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"strategy": {"adx_treshold": 20}}))  # typo
    with pytest.raises(ValueError, match="Unknown key"):
        Config.from_yaml(p)


def test_missing_sections_fall_back_to_defaults(tmp_path):
    p = tmp_path / "min.yaml"
    p.write_text(yaml.safe_dump({"data": {"symbol": "^NSEBANK"}}))
    cfg = Config.from_yaml(p)
    assert cfg.data.symbol == "^NSEBANK"
    assert cfg.risk.risk_per_trade_pct == 0.01  # default survived


def test_invalid_trail_mode_is_rejected():
    with pytest.raises(ValueError, match="supertrend_trail_mode"):
        StrategyConfig(supertrend_trail_mode="sometimes")


def test_invalid_entry_mode_is_rejected():
    with pytest.raises(ValueError, match="entry_mode"):
        StrategyConfig(entry_mode="hunch")


def test_round_trip_to_dict():
    cfg = Config()
    d = cfg.to_dict()
    assert d["risk"]["lot_size"] == cfg.risk.lot_size
    assert set(d) == {"data", "session", "strategy", "risk", "costs", "backtest"}


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------
def _write_csv(path, index, df, ts_name="datetime", tz_suffix=""):
    rows = ["%s,open,high,low,close,volume" % ts_name]
    for ts, r in zip(index, df.itertuples()):
        stamp = ts.strftime("%Y-%m-%d %H:%M:%S") + tz_suffix
        rows.append(f"{stamp},{r.open},{r.high},{r.low},{r.close},{r.volume}")
    path.write_text("\n".join(rows))


def test_naive_timestamps_are_treated_as_ist(tmp_path):
    idx = make_session_index(2)
    df = synth_ohlcv(idx)
    p = tmp_path / "bars.csv"
    _write_csv(p, idx, df)

    out = load_from_csv(p, "Asia/Kolkata")
    assert str(out.index.tz) == "Asia/Kolkata"
    assert out.index[0].strftime("%H:%M") == "09:15"
    assert len(out) == len(df)


def test_alternative_timestamp_column_names(tmp_path):
    idx = make_session_index(1)
    df = synth_ohlcv(idx)
    for name in ("date", "timestamp", "time"):
        p = tmp_path / f"{name}.csv"
        _write_csv(p, idx, df, ts_name=name)
        out = load_from_csv(p, "Asia/Kolkata")
        assert len(out) == len(df)


def test_missing_timestamp_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("o,h,l,c\n1,2,0,1\n")
    with pytest.raises(ValueError, match="timestamp column"):
        load_from_csv(p, "Asia/Kolkata")


def test_missing_price_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("datetime,open,high,low\n2024-01-01 09:15:00,1,2,0\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_from_csv(p, "Asia/Kolkata")


def test_bad_bars_are_repaired_not_dropped(tmp_path):
    """A vendor tick where high < close must be fixed, keeping the session whole."""
    p = tmp_path / "bad.csv"
    p.write_text(
        "datetime,open,high,low,close,volume\n"
        "2024-01-01 09:15:00,100,101,99,105,10\n"   # high below close
        "2024-01-01 09:30:00,105,106,110,107,10\n"  # low above close
    )
    out = load_from_csv(p, "Asia/Kolkata")
    assert len(out) == 2
    assert (out["high"] >= out["close"]).all()
    assert (out["low"] <= out["close"]).all()


def test_duplicate_timestamps_are_collapsed(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text(
        "datetime,open,high,low,close,volume\n"
        "2024-01-01 09:15:00,100,101,99,100,10\n"
        "2024-01-01 09:15:00,200,201,199,200,10\n"
    )
    out = load_from_csv(p, "Asia/Kolkata")
    assert len(out) == 1
    assert out["close"].iloc[0] == 100.0  # first one wins


# --------------------------------------------------------------------------
# Session filtering and resampling
# --------------------------------------------------------------------------
def test_bars_outside_the_session_are_dropped(cfg):
    idx = make_session_index(3)
    df = synth_ohlcv(idx)
    stray = df.iloc[[0]].copy()
    stray.index = [idx[0].replace(hour=8, minute=0)]
    polluted = pd.concat([df, stray]).sort_index()

    out = filter_session(polluted, cfg.session, is_intraday=True)
    assert len(out) == len(df)
    assert out.index.time.min() >= pd.Timestamp("09:15").time()


def test_daily_data_is_not_session_filtered(cfg):
    idx = pd.date_range("2024-01-01", periods=50, freq="B", tz="Asia/Kolkata")
    df = synth_ohlcv(idx)
    assert len(filter_session(df, cfg.session, is_intraday=False)) == len(df)


@pytest.mark.parametrize(
    "interval,expected",
    [("1m", True), ("5m", True), ("15m", True), ("1h", True),
     ("1d", False), ("1wk", False), ("1mo", False)],
)
def test_interval_classification(interval, expected):
    assert is_intraday_interval(interval) is expected


def test_resample_aggregates_the_right_window():
    idx = make_session_index(2)
    df = synth_ohlcv(idx)
    out = resample_htf(df, 4)
    i = 10
    assert out["high"].iloc[i] == pytest.approx(df["high"].iloc[i - 3:i + 1].max())
    assert out["low"].iloc[i] == pytest.approx(df["low"].iloc[i - 3:i + 1].min())
    assert out["close"].iloc[i] == pytest.approx(df["close"].iloc[i])
    assert out["open"].iloc[i] == pytest.approx(df["open"].iloc[i - 3])


def test_resample_of_one_is_the_identity():
    df = synth_ohlcv(make_session_index(2))
    pd.testing.assert_frame_equal(resample_htf(df, 1), df)
