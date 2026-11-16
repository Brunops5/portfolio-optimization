"""Unit tests for portfolio backtesting and performance metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from backtester import (
    backtest_strategy,
    calculate_metrics,
    compute_drawdown_series,
    maximum_drawdown,
    run_full_backtest,
)


class TestBacktestStrategy:
    @pytest.fixture
    def simple_data(self):
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")
        number_of_dates = len(dates)
        returns = pd.DataFrame(
            {
                "A": [0.01] * number_of_dates,
                "B": [0.0] * number_of_dates,
            },
            index=dates,
        )
        weights = pd.DataFrame(
            {"A": [0.5], "B": [0.5]},
            index=[dates[0]],
        )
        return returns, weights

    def test_portfolio_grows_with_positive_returns(self, simple_data):
        returns, weights = simple_data

        result = backtest_strategy(
            returns,
            weights,
            transaction_cost=0.0,
            initial_capital=1000.0,
        )

        final_value = result["portfolio_value"].iloc[-1]
        first_value = result["portfolio_value"].iloc[0]
        assert final_value > first_value

    def test_transaction_cost_reduces_value(self):
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        weights = pd.DataFrame(
            {"A": [0.5, 0.8], "B": [0.5, 0.2]},
            index=[dates[0], dates[3]],
        )

        without_cost = backtest_strategy(
            returns,
            weights,
            transaction_cost=0.0,
            initial_capital=1000.0,
        )
        with_cost = backtest_strategy(
            returns,
            weights,
            transaction_cost=0.01,
            initial_capital=1000.0,
        )

        assert (
            with_cost["portfolio_value"].iloc[-1]
            < without_cost["portfolio_value"].iloc[-1]
        )

    def test_transaction_cost_is_charged_once(self):
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        weights = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            index=[dates[0], dates[1]],
        )

        result = backtest_strategy(
            returns,
            weights,
            transaction_cost=0.01,
            initial_capital=1000,
        )

        assert result.loc[dates[1], "portfolio_value"] == pytest.approx(980.0)
        assert result.loc[dates[1], "transaction_cost"] == pytest.approx(20.0)

    def test_weights_drift_between_rebalances(self):
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        returns = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 0.0]},
            index=dates,
        )
        weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[dates[0]])

        result = backtest_strategy(
            returns,
            weights,
            transaction_cost=0,
            initial_capital=100,
        )

        assert result.iloc[0]["portfolio_value"] == pytest.approx(150.0)
        assert result.iloc[1]["portfolio_value"] == pytest.approx(150.0)

    def test_missing_held_return_preserves_last_value_and_is_reported(self):
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        returns = pd.DataFrame(
            {"A": [0.1, np.nan], "B": [0.0, 0.0]},
            index=dates,
        )
        weights = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[dates[0]])

        result = backtest_strategy(
            returns,
            weights,
            transaction_cost=0,
            initial_capital=100,
        )

        assert result.iloc[-1]["portfolio_value"] == pytest.approx(110.0)
        assert result.iloc[-1]["stale_positions"] == 1

    def test_missing_position_on_rebalance_is_reported_before_liquidation(self):
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        returns = pd.DataFrame(
            {"A": [0.0, np.nan], "B": [0.0, 0.0]},
            index=dates,
        )
        weights = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            index=dates,
        )

        result = backtest_strategy(
            returns,
            weights,
            transaction_cost=0,
            initial_capital=100,
        )

        assert result.loc[dates[1], "stale_positions"] == 1

    def test_invalid_weight_sum_is_rejected(self):
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="B")
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        weights = pd.DataFrame({"A": [0.6], "B": [0.6]}, index=[dates[0]])

        with pytest.raises(ValueError, match="sum to 1"):
            backtest_strategy(returns, weights, transaction_cost=0.0)

    def test_full_backtest_collects_trading_metrics(self):
        dates = pd.date_range("2023-01-02", periods=14, freq="D")
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        weights = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            index=[dates[0], dates[7]],
        )

        _, metrics = run_full_backtest(
            returns,
            {"strategy": weights},
            transaction_cost=0.01,
            initial_capital=1000.0,
        )

        assert metrics.loc["strategy", "total_turnover"] == pytest.approx(2.0)
        assert metrics.loc["strategy", "total_transaction_cost"] == pytest.approx(20.0)
        assert metrics.loc["strategy", "rebalance_events"] == 1


class TestDrawdown:
    def test_no_drawdown_for_monotonic_growth(self):
        values = pd.Series(
            [100, 110, 120, 130],
            index=pd.date_range("2023-01-01", periods=4),
        )

        drawdown = compute_drawdown_series(values)

        assert drawdown.min() == pytest.approx(0.0)

    def test_drawdown_after_peak(self):
        values = pd.Series(
            [100, 120, 90, 110],
            index=pd.date_range("2023-01-01", periods=4),
        )

        drawdown = compute_drawdown_series(values)

        assert drawdown.min() == pytest.approx(-0.25, abs=1e-6)

    def test_maximum_drawdown_value(self):
        values = pd.Series(
            [100, 120, 80, 100],
            index=pd.date_range("2023-01-01", periods=4),
        )

        assert maximum_drawdown(values) == pytest.approx(-1 / 3, abs=1e-6)

    def test_drawdown_includes_initial_capital(self):
        values = pd.Series(
            [90.0, 95.0],
            index=pd.date_range("2023-01-01", periods=2),
        )

        drawdown = maximum_drawdown(values, initial_capital=100.0)

        assert drawdown == pytest.approx(-0.1)


class TestCalculateMetrics:
    def test_metrics_structure(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        values = pd.Series(
            10000 * (1 + np.linspace(0, 1, len(dates))),
            index=dates,
        )
        daily_returns = pd.Series(
            np.log(values / values.shift(1)).dropna(),
            index=dates[1:],
        )

        metrics = calculate_metrics(values, daily_returns)

        expected_keys = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
        }
        assert expected_keys == set(metrics)

    def test_zero_volatility_handled(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        values = pd.Series([10000.0] * len(dates), index=dates)
        daily_returns = pd.Series([0.0] * (len(dates) - 1), index=dates[1:])

        metrics = calculate_metrics(values, daily_returns)

        assert metrics["annualized_volatility"] == pytest.approx(0.0)

    def test_positive_total_return(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        values = pd.Series(
            10000 * np.exp(np.linspace(0, 0.5, len(dates))),
            index=dates,
        )
        daily_returns = pd.Series(
            np.log(values / values.shift(1)).dropna(),
            index=dates[1:],
        )

        metrics = calculate_metrics(values, daily_returns)

        assert metrics["total_return"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
