"""Unit tests for periodic returns and Wilcoxon comparisons."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from stats_tests import periodic_returns_from_backtest, wilcoxon_comparison


class TestPeriodicReturnsFromBacktest:
    def test_log_returns_consistency(self):
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
        values = np.exp(np.linspace(0, 1.5, len(dates)))
        backtest = pd.DataFrame(
            {
                "portfolio_value": values,
                "daily_return": pd.Series(values, index=dates)
                .pct_change(fill_method=None)
                .fillna(0.0),
            },
            index=dates,
        )

        monthly = periodic_returns_from_backtest(backtest, frequency="ME")

        assert len(monthly) == 3
        assert np.all(np.isfinite(monthly))

    def test_matches_manual_calculation(self):
        dates = pd.date_range("2023-01-01", "2023-04-30", freq="B")
        values = np.linspace(100, 200, len(dates))
        backtest = pd.DataFrame(
            {
                "portfolio_value": values,
                "daily_return": pd.Series(values, index=dates)
                .pct_change(fill_method=None)
                .fillna(0.0),
            },
            index=dates,
        )

        monthly = periodic_returns_from_backtest(backtest, frequency="ME")

        january_end = backtest["portfolio_value"].resample("ME").last().iloc[0]
        february_end = backtest["portfolio_value"].resample("ME").last().iloc[1]
        expected = np.log(february_end / january_end)
        assert monthly.loc[pd.Timestamp("2023-02-28")] == pytest.approx(expected)

    def test_preserves_first_month(self):
        dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
        backtest = pd.DataFrame({"daily_return": 0.01}, index=dates)

        monthly = periodic_returns_from_backtest(backtest, frequency="ME")

        assert list(monthly.index) == [
            pd.Timestamp("2023-01-31"),
            pd.Timestamp("2023-02-28"),
        ]
        assert monthly.iloc[0] == pytest.approx(31 * np.log(1.01))

    def test_weekly_non_overlapping_returns(self):
        dates = pd.date_range("2023-01-02", periods=14, freq="D")
        backtest = pd.DataFrame({"daily_return": 0.01}, index=dates)

        weekly = periodic_returns_from_backtest(backtest, frequency="W-SUN")

        assert list(weekly.index) == [
            pd.Timestamp("2023-01-08"),
            pd.Timestamp("2023-01-15"),
        ]
        assert np.allclose(weekly.to_numpy(), 7 * np.log(1.01))

    def test_weekly_drops_partial_last_week(self):
        dates = pd.date_range("2023-01-02", periods=10, freq="D")
        backtest = pd.DataFrame({"daily_return": 0.01}, index=dates)

        weekly = periodic_returns_from_backtest(backtest, frequency="W-SUN")

        assert list(weekly.index) == [pd.Timestamp("2023-01-08")]


class TestWilcoxonComparison:
    def test_identical_series(self):
        dates = pd.date_range("2023-01", "2023-12", freq="ME")
        returns = pd.Series(
            np.random.default_rng(42).normal(0.02, 0.05, len(dates)),
            index=dates,
        )

        stats = wilcoxon_comparison(returns, returns, label_a="A", label_b="B")

        assert np.isnan(stats["win_rate_a"])
        assert np.isnan(stats["win_rate_b"])
        assert stats["mean_diff"] == pytest.approx(0.0, abs=1e-12)

    def test_clearly_different_series(self):
        dates = pd.date_range("2023-01", "2023-12", freq="ME")
        returns_a = pd.Series([0.05] * len(dates), index=dates)
        returns_b = pd.Series([-0.05] * len(dates), index=dates)

        stats = wilcoxon_comparison(returns_a, returns_b, label_a="A", label_b="B")

        assert stats["p_value"] < 0.01
        assert stats["win_rate_a"] == pytest.approx(1.0)
        assert stats["win_rate_b"] == pytest.approx(0.0)

    def test_insufficient_data(self):
        dates = pd.date_range("2023-01", "2023-03", freq="ME")
        returns_a = pd.Series([0.01, 0.02][: len(dates)], index=dates)
        returns_b = pd.Series([0.02, 0.01][: len(dates)], index=dates)

        stats = wilcoxon_comparison(returns_a, returns_b, label_a="A", label_b="B")

        assert np.isnan(stats["p_value"])

    def test_stats_keys(self):
        dates = pd.date_range("2023-01", "2023-12", freq="ME")
        generator = np.random.default_rng(42)
        returns_a = pd.Series(
            generator.normal(0.02, 0.05, len(dates)),
            index=dates,
        )
        returns_b = pd.Series(
            generator.normal(0.01, 0.05, len(dates)),
            index=dates,
        )

        stats = wilcoxon_comparison(returns_a, returns_b)

        expected_keys = {
            "n",
            "wilcoxon_statistic",
            "p_value",
            "median_diff",
            "mean_diff",
            "std_diff",
            "min_diff",
            "max_diff",
            "win_rate_a",
            "win_rate_b",
        }
        assert expected_keys.issubset(stats.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
