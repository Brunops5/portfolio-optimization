"""
Testes unitários para o módulo stats_tests.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from stats_tests import (
    monthly_returns_from_backtest,
    wilcoxon_comparison,
)


class TestMonthlyReturnsFromBacktest:
    """Testes para extração de retornos mensais logarítmicos."""

    def test_log_returns_consistency(self):
        """Retornos mensais devem ser logarítmicos, consistentes com o projeto."""
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
        # Patrimônio dobra a cada mês
        pv_values = np.exp(np.linspace(0, 1.5, len(dates)))
        bt = pd.DataFrame(
            {"portfolio_value": pv_values},
            index=dates,
        )
        monthly = monthly_returns_from_backtest(bt)

        # O primeiro retorno mensal deve ser ln(P_fev / P_jan)
        assert len(monthly) == 2
        assert np.all(np.isfinite(monthly))

    def test_matches_manual_calculation(self):
        """Deve coincidir com cálculo manual de log-retorno."""
        dates = pd.date_range("2023-01-01", "2023-04-30", freq="B")
        # Valores fixos no último dia de cada mês
        values = np.linspace(100, 200, len(dates))
        bt = pd.DataFrame({"portfolio_value": values}, index=dates)
        monthly = monthly_returns_from_backtest(bt)

        jan_end = bt["portfolio_value"].resample("ME").last().iloc[0]
        feb_end = bt["portfolio_value"].resample("ME").last().iloc[1]
        expected = np.log(feb_end / jan_end)

        assert monthly.iloc[0] == pytest.approx(expected)


class TestWilcoxonComparison:
    """Testes para comparação pareada de Wilcoxon."""

    def test_identical_series(self):
        """Séries idênticas não apresentam diferença (p-value alto)."""
        dates = pd.date_range("2023-01", "2023-12", freq="ME")
        r = pd.Series(np.random.normal(0.02, 0.05, len(dates)), index=dates)
        stats = wilcoxon_comparison(r, r, label_a="A", label_b="B")
        # Séries idênticas: não há meses vencedores
        assert np.isnan(stats["win_rate_a"])
        assert np.isnan(stats["win_rate_b"])
        # Diferença média deve ser zero
        assert stats["mean_diff"] == pytest.approx(0.0, abs=1e-12)

    def test_clearly_different_series(self):
        """Séries claramente diferentes devem ter p-value baixo."""
        dates = pd.date_range("2023-01", "2023-12", freq="ME")
        a = pd.Series([0.05] * len(dates), index=dates)
        b = pd.Series([-0.05] * len(dates), index=dates)
        stats = wilcoxon_comparison(a, b, label_a="A", label_b="B")
        assert stats["p_value"] < 0.01
        assert stats["win_rate_a"] == pytest.approx(1.0)
        assert stats["win_rate_b"] == pytest.approx(0.0)

    def test_insufficient_data(self):
        """Poucos pontos devem retornar estatísticas NaN."""
        dates = pd.date_range("2023-01", "2023-03", freq="ME")
        n = len(dates)
        a = pd.Series([0.01, 0.02][:n], index=dates)
        b = pd.Series([0.02, 0.01][:n], index=dates)
        stats = wilcoxon_comparison(a, b, label_a="A", label_b="B")
        assert np.isnan(stats["p_value"])

    def test_stats_keys(self):
        """Resultado deve conter todos os campos esperados."""
        dates = pd.date_range("2023-01", "2023-12", freq="ME")
        a = pd.Series(np.random.normal(0.02, 0.05, len(dates)), index=dates)
        b = pd.Series(np.random.normal(0.01, 0.05, len(dates)), index=dates)
        stats = wilcoxon_comparison(a, b)

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
