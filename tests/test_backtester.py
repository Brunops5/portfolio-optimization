"""
Testes unitários para o módulo backtester.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from backtester import (
    backtest_strategy,
    compute_drawdown_series,
    maximum_drawdown,
    calculate_metrics,
    run_full_backtest,
)


class TestBacktestStrategy:
    """Testes para a função de backtest."""

    @pytest.fixture
    def simple_data(self):
        """Dados determinísticos simples para validação manual."""
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")
        n = len(dates)
        rets = pd.DataFrame(
            {
                "A": [0.01] * n,
                "B": [0.0] * n,
            },
            index=dates,
        )
        weights = pd.DataFrame(
            {"A": [0.5], "B": [0.5]},
            index=[dates[0]],
        )
        return rets, weights

    def test_portfolio_grows_with_positive_returns(self, simple_data):
        """Carteira com retornos positivos diários deve crescer."""
        rets, weights = simple_data
        bt = backtest_strategy(rets, weights, transaction_cost=0.0, initial_capital=1000.0)
        assert bt["portfolio_value"].iloc[-1] > bt["portfolio_value"].iloc[0]

    def test_transaction_cost_reduces_value(self):
        """Custos de transação devem reduzir o patrimônio final com turnover."""
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")
        n = len(dates)
        # Retornos nulos para isolar o efeito do custo
        rets = pd.DataFrame({"A": [0.0] * n, "B": [0.0] * n}, index=dates)
        # Dois rebalanceamentos com pesos diferentes → gera turnover
        weights = pd.DataFrame(
            {"A": [0.5, 0.8], "B": [0.5, 0.2]},
            index=[dates[0], dates[3]],
        )
        bt_no_cost = backtest_strategy(rets, weights, transaction_cost=0.0, initial_capital=1000.0)
        bt_with_cost = backtest_strategy(
            rets, weights, transaction_cost=0.01, initial_capital=1000.0
        )
        assert bt_with_cost["portfolio_value"].iloc[-1] < bt_no_cost["portfolio_value"].iloc[-1]

    def test_transaction_cost_is_charged_once(self):
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        rets = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        weights = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[dates[0], dates[1]]
        )
        result = backtest_strategy(rets, weights, transaction_cost=0.01, initial_capital=1000)
        assert result.loc[dates[1], "portfolio_value"] == pytest.approx(980.0)
        assert result.loc[dates[1], "transaction_cost"] == pytest.approx(20.0)

    def test_weights_drift_between_rebalances(self):
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        rets = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 0.0]}, index=dates)
        weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[dates[0]])
        result = backtest_strategy(rets, weights, transaction_cost=0, initial_capital=100)
        assert result.iloc[0]["portfolio_value"] == pytest.approx(150.0)
        assert result.iloc[1]["portfolio_value"] == pytest.approx(150.0)

    def test_missing_held_return_preserves_last_value_and_is_reported(self):
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        rets = pd.DataFrame({"A": [0.1, np.nan], "B": [0.0, 0.0]}, index=dates)
        weights = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[dates[0]])
        result = backtest_strategy(rets, weights, transaction_cost=0, initial_capital=100)
        assert result.iloc[-1]["portfolio_value"] == pytest.approx(110.0)
        assert result.iloc[-1]["stale_positions"] == 1

    def test_missing_position_on_rebalance_is_reported_before_liquidation(self):
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        rets = pd.DataFrame(
            {"A": [0.0, np.nan], "B": [0.0, 0.0]}, index=dates
        )
        weights = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=dates
        )

        result = backtest_strategy(
            rets, weights, transaction_cost=0, initial_capital=100
        )

        assert result.loc[dates[1], "stale_positions"] == 1

    def test_invalid_weight_sum_is_rejected(self):
        """O backtest não deve corrigir silenciosamente uma carteira inválida."""
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="B")
        n = len(dates)
        rets = pd.DataFrame({"A": [0.0] * n, "B": [0.0] * n}, index=dates)
        weights = pd.DataFrame({"A": [0.6], "B": [0.6]}, index=[dates[0]])  # soma 1.2
        with pytest.raises(ValueError, match="somar 1"):
            backtest_strategy(rets, weights, transaction_cost=0.0)

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
    """Testes para cálculo de drawdown."""

    def test_no_drawdown_for_monotonic_growth(self):
        """Série estritamente crescente não apresenta drawdown."""
        pv = pd.Series([100, 110, 120, 130], index=pd.date_range("2023-01-01", periods=4))
        dd = compute_drawdown_series(pv)
        assert dd.min() == pytest.approx(0.0)

    def test_drawdown_after_peak(self):
        """Drawdown deve ser negativo após queda do pico."""
        pv = pd.Series([100, 120, 90, 110], index=pd.date_range("2023-01-01", periods=4))
        dd = compute_drawdown_series(pv)
        assert dd.min() == pytest.approx(-0.25, abs=1e-6)

    def test_maximum_drawdown_value(self):
        """MDD deve corresponder ao valor mínimo da série de drawdown."""
        pv = pd.Series([100, 120, 80, 100], index=pd.date_range("2023-01-01", periods=4))
        assert maximum_drawdown(pv) == pytest.approx(-1 / 3, abs=1e-6)

    def test_drawdown_includes_initial_capital(self):
        pv = pd.Series([90.0, 95.0], index=pd.date_range("2023-01-01", periods=2))
        assert maximum_drawdown(pv, initial_capital=100.0) == pytest.approx(-0.1)


class TestCalculateMetrics:
    """Testes para métricas de desempenho."""

    def test_metrics_structure(self):
        """Métricas devem conter todos os campos esperados."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        n = len(dates)
        # Carteira que dobra em 1 ano
        pv = pd.Series(10000 * (1 + np.linspace(0, 1, n)), index=dates)
        daily_rets = pd.Series(np.log(pv / pv.shift(1)).dropna(), index=dates[1:])
        metrics = calculate_metrics(pv, daily_rets)

        expected_keys = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
        }
        assert expected_keys == set(metrics)

    def test_zero_volatility_handled(self):
        """Carteira constante deve registrar volatilidade igual a zero."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        pv = pd.Series([10000.0] * len(dates), index=dates)
        daily_rets = pd.Series([0.0] * (len(dates) - 1), index=dates[1:])
        metrics = calculate_metrics(pv, daily_rets)
        assert metrics["annualized_volatility"] == pytest.approx(0.0)

    def test_positive_total_return(self):
        """Carteira que cresce deve ter retorno total positivo."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        n = len(dates)
        pv = pd.Series(10000 * np.exp(np.linspace(0, 0.5, n)), index=dates)
        daily_rets = pd.Series(np.log(pv / pv.shift(1)).dropna(), index=dates[1:])
        metrics = calculate_metrics(pv, daily_rets)
        assert metrics["total_return"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
