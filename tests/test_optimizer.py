"""
Testes unitários para o módulo optimizer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Adiciona src/ ao path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from optimizer import weighted_average_correlation, optimize_portfolio


class TestWeightedAverageCorrelation:
    """Testes para a função de correlação média ponderada."""

    def test_perfect_correlation(self):
        """Se todos os pares têm correlação 1, a média deve ser 1."""
        corr = np.ones((3, 3))
        weights = np.array([0.5, 0.3, 0.2])
        assert weighted_average_correlation(weights, corr) == pytest.approx(1.0)

    def test_zero_correlation_off_diagonal(self):
        """Correlação zero fora da diagonal deve resultar em 0."""
        corr = np.eye(3)
        weights = np.array([0.5, 0.3, 0.2])
        assert weighted_average_correlation(weights, corr) == pytest.approx(0.0)

    def test_two_assets_equal_weights(self):
        """Dois ativos com correlação ρ e pesos iguais: média = ρ."""
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        weights = np.array([0.5, 0.5])
        assert weighted_average_correlation(weights, corr) == pytest.approx(0.6)

    def test_single_asset_returns_zero(self):
        """Carteira com um único ativo não tem pares: correlação = 0."""
        corr = np.ones((1, 1))
        weights = np.array([1.0])
        assert weighted_average_correlation(weights, corr) == pytest.approx(0.0)

    def test_weights_sum_less_than_one(self):
        """Pesos não normalizados não devem afetar a média ponderada."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        weights = np.array([1.0, 1.0])  # não soma 1, mas a fração se cancela
        assert weighted_average_correlation(weights, corr) == pytest.approx(0.5)


class TestOptimizePortfolio:
    """Testes para a função principal de otimização."""

    @pytest.fixture
    def returns(self):
        """Retornos diários sintéticos para 3 ativos."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", "2022-12-31", freq="B")
        return pd.DataFrame(
            np.random.normal(0.001, 0.02, (len(dates), 3)),
            index=dates,
            columns=["BTC", "ETH", "SOL"],
        )

    def test_weights_sum_to_one(self, returns):
        """Os pesos otimizados devem somar 1."""
        weights, info = optimize_portfolio(returns, max_avg_correlation=0.85)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self, returns):
        """Não deve haver pesos negativos (short selling proibido)."""
        weights, info = optimize_portfolio(returns, max_avg_correlation=0.85)
        assert np.all(weights >= -1e-8)

    def test_weights_within_bounds(self, returns):
        """Todos os pesos devem respeitar os bounds [min, max]."""
        weights, info = optimize_portfolio(
            returns,
            max_avg_correlation=0.85,
            min_weight=0.0,
            max_weight=0.40,
        )
        assert np.all(weights <= 0.40 + 1e-8)
        assert np.all(weights >= 0.0 - 1e-8)

    def test_info_has_expected_keys(self, returns):
        """O dict de informações deve conter as métricas esperadas."""
        weights, info = optimize_portfolio(returns, max_avg_correlation=0.85)
        expected_keys = {
            "success",
            "message",
            "expected_daily_return",
            "expected_annual_return",
            "daily_volatility",
            "annual_volatility",
            "sharpe_ratio",
            "avg_correlation",
            "num_assets_selected",
        }
        assert expected_keys.issubset(info.keys())

    def test_sharpe_ratio_sensible(self, returns):
        """Sharpe Ratio deve ser um número finito."""
        weights, info = optimize_portfolio(returns, max_avg_correlation=0.85)
        assert np.isfinite(info["sharpe_ratio"])

    def test_conservative_may_fallback(self, returns):
        """Perfil conservador pode fazer fallback se restrição for inviável."""
        # Dados aleatórios provavelmente não violam 0.99, então otimiza normalmente
        weights, info = optimize_portfolio(returns, max_avg_correlation=0.99)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
