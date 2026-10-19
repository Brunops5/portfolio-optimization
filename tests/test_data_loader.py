"""Testes de preparação e cache de dados."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import data_loader
from data_loader import (
    chunk_date_range,
    compute_returns,
    download_asset,
    load_all_assets,
)


def test_simple_and_log_returns():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0]},
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
    )
    result = compute_returns(prices)
    assert result["daily_simple"]["A"].iloc[0] == pytest.approx(0.1)
    assert set(result) == {"prices", "daily_simple"}


def test_no_backward_or_post_history_fill():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "A": [np.nan, 100.0, np.nan, 110.0, np.nan],
            "B": [50.0, 51.0, 52.0, 53.0, 54.0],
        },
        index=dates,
    )
    filled = compute_returns(prices)["prices"]["A"]
    assert pd.isna(filled.iloc[0])
    assert filled.iloc[2] == 100.0
    assert pd.isna(filled.iloc[4])


def test_legacy_coinbase_source_is_inferred_from_original_schema():
    legacy = pd.DataFrame(
        {
            "timestamp": ["2023-01-01"],
            "low": [1.0],
            "high": [2.0],
            "open": [1.5],
            "close": [1.8],
            "volume": [10.0],
            "source": ["legacy"],
        }
    )

    result = data_loader._normalise_market_data(legacy)

    assert result.iloc[0]["source"] == "coinbase"
    assert result.iloc[0]["source_status"] == "inferred_from_legacy_schema"


def test_ambiguous_legacy_source_is_rejected():
    legacy = pd.DataFrame(
        {"timestamp": ["2023-01-01"], "close": [1.0], "source": ["legacy"]}
    )

    with pytest.raises(ValueError, match="fonte não pode ser inferida"):
        data_loader._normalise_market_data(legacy)


def test_long_internal_gap_is_not_turned_into_observations(monkeypatch):
    monkeypatch.setitem(
        data_loader.CONFIG["optimization"], "max_internal_gap_days", 2
    )
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"A": [100.0, *([np.nan] * 8), 110.0]}, index=dates)

    result = compute_returns(prices)

    assert result["prices"]["A"].notna().sum() == 4
    assert result["daily_simple"]["A"].notna().sum() == 2


def test_chunks_do_not_overlap():
    chunks = chunk_date_range("2023-01-01", "2024-01-01", max_days=100)
    for (_, previous_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert pd.Timestamp(next_start) == pd.Timestamp(previous_end) + pd.Timedelta(days=1)


def test_partial_coinbase_history_does_not_fall_back_to_same_ticker(monkeypatch):
    """Um lançamento recente não pode ser trocado por homônimo do Yahoo."""
    coinbase_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-03-23", periods=2, freq="D"),
            "close": [1.20, 1.25],
            "source": ["coinbase", "coinbase"],
        }
    )
    monkeypatch.setattr(data_loader, "fetch_candles", lambda *args, **kwargs: coinbase_data)
    monkeypatch.setattr(data_loader.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        data_loader,
        "fetch_yfinance",
        lambda *args, **kwargs: pytest.fail("não deve consultar ticker homônimo"),
    )

    result = data_loader._download_interval("ARB", "2023-03-20", "2023-03-31")

    assert len(result) == 2
    assert set(result["source"]) == {"coinbase"}


def test_coinbase_cache_does_not_use_yahoo_for_prelaunch_interval(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PORTFOLIO_CACHE_DIR", str(tmp_path))
    cache_file = tmp_path / "market-data" / "ARB-USD_86400.csv"
    cache_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-03-23", periods=2, freq="D"),
            "close": [1.20, 1.25],
            "source": ["coinbase", "coinbase"],
        }
    ).to_csv(cache_file, index=False)

    def empty_prelaunch(asset, start, end, allow_yahoo=True):
        assert not allow_yahoo
        return pd.DataFrame()

    monkeypatch.setattr(data_loader, "_download_interval", empty_prelaunch)

    result = download_asset("ARB", "2023-03-20", "2023-03-24")

    assert len(result) == 2
    assert set(result["source"]) == {"coinbase"}


def test_complete_cache_avoids_download(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_CACHE_DIR", str(tmp_path))
    cache_file = tmp_path / "market-data" / "BTC-USD_86400.csv"
    cache_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", "2023-01-03", freq="D"),
            "close": [1, 2, 3],
            "source": ["test"] * 3,
        }
    ).to_csv(cache_file, index=False)
    monkeypatch.setattr(
        data_loader,
        "_download_interval",
        lambda *args, **kwargs: pytest.fail("não deveria acessar a rede"),
    )
    result = download_asset("BTC", "2023-01-01", "2023-01-03")
    assert len(result) == 3


def test_internal_cache_gap_is_downloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_CACHE_DIR", str(tmp_path))
    cache_file = tmp_path / "market-data" / "BTC-USD_86400.csv"
    cache_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": ["2023-01-01", "2023-01-03"],
            "close": [1.0, 3.0],
            "source": ["test", "test"],
        }
    ).to_csv(cache_file, index=False)

    downloaded = pd.DataFrame(
        {"timestamp": ["2023-01-02"], "close": [2.0], "source": ["test"]}
    )
    calls = []

    def download(asset, start, end, allow_yahoo=True):
        calls.append((start, end))
        return downloaded

    monkeypatch.setattr(data_loader, "_download_interval", download)
    result = download_asset("BTC", "2023-01-01", "2023-01-03")

    assert calls == [("2023-01-02", "2023-01-02")]
    assert len(result) == 3


def test_coinbase_not_found_uses_yahoo(monkeypatch):
    class NotFoundResponse:
        status_code = 404

    expected = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2023-01-01")], "close": [1.0]}
    )
    monkeypatch.setattr(data_loader.requests, "get", lambda *args, **kwargs: NotFoundResponse())
    monkeypatch.setattr(data_loader.time, "sleep", lambda *_: None)
    monkeypatch.setattr(data_loader, "fetch_yfinance", lambda *args, **kwargs: expected)

    result = data_loader._download_interval("UNKNOWN", "2023-01-01", "2023-01-02")

    pd.testing.assert_frame_equal(result, expected)


def test_failed_cache_expansion_is_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_CACHE_DIR", str(tmp_path))
    cache_file = tmp_path / "market-data" / "BTC-USD_86400.csv"
    cache_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", "2023-01-03", freq="D"),
            "close": [1, 2],
            "source": ["test", "test"],
        }
    ).to_csv(cache_file, index=False)
    calls = []

    def transient_failure(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("falha transitória")

    monkeypatch.setattr(data_loader, "_download_interval", transient_failure)
    download_asset("BTC", "2023-01-01", "2023-01-04")
    calls_after_first_run = len(calls)
    download_asset("BTC", "2023-01-01", "2023-01-04")

    assert calls_after_first_run == 2
    assert len(calls) == 4


def test_asset_failure_stops_incomplete_experiment(monkeypatch):
    def fail_one_asset(asset, *args, **kwargs):
        if asset == "ETH":
            raise RuntimeError("fonte indisponível")
        return pd.DataFrame(
            {"timestamp": pd.date_range("2023-01-01", periods=2), "close": [1, 2]}
        )

    monkeypatch.setattr(data_loader, "download_asset", fail_one_asset)

    with pytest.raises(RuntimeError, match="ETH"):
        load_all_assets(["BTC", "ETH"])
