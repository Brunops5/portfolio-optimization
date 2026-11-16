"""Unit tests for mathematical and financial validators."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from validators import (
    validate_backtest_consistency,
    validate_historical_constituents,
    validate_metrics,
    validate_weights,
)


class TestValidateWeights:
    def test_valid_weights_pass(self):
        weights = pd.DataFrame(
            {"A": [0.4, 0.3], "B": [0.3, 0.3], "C": [0.3, 0.4]},
            index=pd.date_range("2023-01", periods=2, freq="ME"),
        )

        errors, warnings = validate_weights({"test": weights})

        assert not errors
        assert not warnings

    def test_weights_not_summing_to_one(self):
        weights = pd.DataFrame(
            {"A": [0.3], "B": [0.5]},
            index=[pd.Timestamp("2023-01-31")],
        )

        errors, _ = validate_weights({"test": weights})

        assert errors

    def test_negative_weights(self):
        weights = pd.DataFrame(
            {"A": [-0.2], "B": [1.2]},
            index=[pd.Timestamp("2023-01-31")],
        )

        errors, _ = validate_weights({"test": weights})

        assert any("below the minimum" in error for error in errors)

    def test_full_allocation_is_allowed_without_cap(self):
        weights = pd.DataFrame(
            {"A": [1.0], "B": [0.0]},
            index=[pd.Timestamp("2023-01-31")],
        )

        errors, _ = validate_weights({"test": weights})

        assert not errors

    def test_asset_outside_historical_composition_is_rejected(self):
        weights = pd.DataFrame(
            {"BTC": [0.5], "MATIC": [0.5]},
            index=[pd.Timestamp("2025-01-01")],
        )

        errors, _ = validate_historical_constituents({"test": weights})

        assert any("MATIC" in error for error in errors)


class TestValidateBacktest:
    @staticmethod
    def _make_backtest(values):
        dates = pd.date_range("2023-01-01", periods=len(values), freq="B")
        portfolio_value = pd.Series(values, index=dates)
        daily_returns = portfolio_value.pct_change(fill_method=None).fillna(0)
        return pd.DataFrame(
            {
                "portfolio_value": portfolio_value,
                "daily_return": daily_returns,
            },
            index=dates,
        )

    def test_positive_portfolio_value(self):
        backtest = self._make_backtest(np.linspace(10000, 11000, 10))
        daily_returns = pd.DataFrame(index=backtest.index)

        errors, _ = validate_backtest_consistency(
            {"test": backtest},
            {"test": pd.DataFrame()},
            daily_returns,
        )

        assert not errors

    def test_negative_portfolio_value(self):
        backtest = self._make_backtest([10000, 5000, -100, 2000, 3000])
        daily_returns = pd.DataFrame(index=backtest.index)

        errors, _ = validate_backtest_consistency(
            {"test": backtest},
            {"test": pd.DataFrame()},
            daily_returns,
        )

        assert any("zero or negative" in error for error in errors)

    def test_stale_position_is_an_error(self):
        backtest = self._make_backtest([10000, 10000])
        backtest["stale_positions"] = [0, 1]
        daily_returns = pd.DataFrame(index=backtest.index)

        errors, warnings = validate_backtest_consistency(
            {"test": backtest},
            {"test": pd.DataFrame()},
            daily_returns,
        )

        assert any("without a new price" in error for error in errors)
        assert not warnings


class TestValidateMetrics:
    def test_valid_metrics(self):
        metrics = pd.DataFrame(
            {
                "total_return": [0.5],
                "annualized_return": [0.2],
                "annualized_volatility": [0.3],
                "max_drawdown": [-0.2],
            },
            index=["test"],
        )

        errors, warnings = validate_metrics(metrics)

        assert not errors
        assert not warnings

    def test_zero_volatility(self):
        metrics = pd.DataFrame(
            {
                "total_return": [0.5],
                "annualized_return": [0.2],
                "annualized_volatility": [0.0],
                "max_drawdown": [-0.2],
            },
            index=["test"],
        )

        errors, _ = validate_metrics(metrics)

        assert any("volatility is near zero" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
