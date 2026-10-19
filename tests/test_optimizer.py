"""Testes da otimização de Markowitz por volatilidade-alvo."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pypfopt import expected_returns, risk_models

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from optimizer import (
    constituent_record_for_date,
    estimate_inputs,
    load_constituent_history,
    optimize_portfolio,
    run_rolling_optimization,
)


@pytest.fixture
def returns():
    generator = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    return pd.DataFrame(
        generator.normal([0.0002, 0.0005, 0.0008], [0.015, 0.025, 0.04], (len(dates), 3)),
        index=dates,
        columns=["BTC", "ETH", "SOL"],
    )


def test_estimate_inputs_matches_lib_defaults(returns):
    mu, covariance, _ = estimate_inputs(returns)

    expected_mu = expected_returns.mean_historical_return(
        returns,
        returns_data=True,
        frequency=365,
    )
    expected_covariance = risk_models.sample_cov(
        returns,
        returns_data=True,
        frequency=365,
    )

    pd.testing.assert_series_equal(mu, expected_mu)
    pd.testing.assert_frame_equal(covariance, expected_covariance)


def test_covariance_is_psd_after_fix(returns):
    _, covariance, info = estimate_inputs(returns)
    assert info["covariance_psd"]
    assert np.linalg.eigvalsh(covariance.values).min() >= -1e-10


def test_weights_and_target_risk(returns):
    weights, info = optimize_portfolio(returns, target_volatility=0.70)
    assert weights.sum() == pytest.approx(1.0, abs=1e-7)
    assert np.all(weights >= -1e-8)
    assert info["target_feasible"]
    assert info["annual_volatility"] <= 0.70 + 1e-5


def test_infeasible_target_is_reported(returns):
    weights, info = optimize_portfolio(returns, target_volatility=0.01)
    assert weights.sum() == pytest.approx(1.0)
    assert not info["target_feasible"]
    assert info["annual_volatility"] == pytest.approx(info["minimum_volatility"], rel=1e-5)


def test_more_risk_cannot_reduce_optimal_expected_return(returns):
    _, lower_risk = optimize_portfolio(returns, target_volatility=0.45)
    _, higher_risk = optimize_portfolio(returns, target_volatility=0.70)

    assert lower_risk["target_feasible"]
    assert higher_risk["target_feasible"]
    assert (
        higher_risk["expected_annual_return"]
        >= lower_risk["expected_annual_return"] - 1e-7
    )


def test_rolling_weights_begin_on_monday_and_have_no_lookahead(returns):
    profiles = {"teste": {"name": "Teste", "target_volatility": 0.9}}
    history = pd.DataFrame(
        {
            "quarter": ["teste"],
            "constituents": [tuple(returns.columns)],
        },
        index=pd.DatetimeIndex(["2021-01-01"], name="effective_date"),
    )
    weights, diagnostics = run_rolling_optimization(
        returns, profiles=profiles, constituent_history=history
    )
    first_date = weights["teste"].index[0]
    assert first_date == pd.Timestamp("2023-01-02")
    assert (weights["teste"].index.weekday == 0).all()
    assert weights["teste"].index.to_series().diff().dropna().eq(pd.Timedelta(days=7)).all()
    assert diagnostics.iloc[0]["observations"] == 365
    assert weights["teste"].index.equals(weights["benchmark"].index)


def test_historical_constituents_are_point_in_time():
    history = load_constituent_history()

    before_june = constituent_record_for_date(history, pd.Timestamp("2024-06-01"))
    after_june = constituent_record_for_date(history, pd.Timestamp("2024-07-01"))
    after_december = constituent_record_for_date(history, pd.Timestamp("2025-01-01"))

    assert "MATIC" not in before_june["constituents"]
    assert "MATIC" in after_june["constituents"]
    assert "MATIC" not in after_december["constituents"]
    assert before_june["quarter"] == "2024-Q1"
    assert after_june["quarter"] == "2024-Q2"
