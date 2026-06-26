"""
Testes unitários para o módulo validators.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from validators import (
    validate_weights,
    validate_backtest_consistency,
    validate_metrics,
)


class TestValidateWeights:
    """Testes para validação de pesos."""

    def test_valid_weights_pass(self):
        """Pesos válidos devem passar sem erros."""
        w = pd.DataFrame(
            {"A": [0.4, 0.3], "B": [0.3, 0.3], "C": [0.3, 0.4]},
            index=pd.date_range("2023-01", periods=2, freq="ME"),
        )
        errors, warnings = validate_weights({"teste": w})
        assert len(errors) == 0

    def test_weights_not_summing_to_one(self):
        """Pesos que não somam 1 devem gerar erro."""
        w = pd.DataFrame({"A": [0.3], "B": [0.5]}, index=[pd.Timestamp("2023-01-31")])
        errors, warnings = validate_weights({"teste": w})
        assert len(errors) > 0

    def test_negative_weights(self):
        """Pesos negativos devem gerar erro."""
        w = pd.DataFrame({"A": [-0.2], "B": [1.2]}, index=[pd.Timestamp("2023-01-31")])
        errors, warnings = validate_weights({"teste": w})
        assert any("negativos" in e for e in errors)

    def test_weight_above_max(self):
        """Peso acima do limite de 40% deve gerar erro."""
        w = pd.DataFrame({"A": [0.9], "B": [0.1]}, index=[pd.Timestamp("2023-01-31")])
        errors, warnings = validate_weights({"teste": w})
        assert any("40%" in e for e in errors)


class TestValidateBacktest:
    """Testes para validação do backtest."""

    def _make_backtest(self, values):
        """Helper que cria backtest com retornos consistentes."""
        n = len(values)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        pv = pd.Series(values, index=dates)
        daily_rets = np.log(pv / pv.shift(1)).fillna(0)
        bt = pd.DataFrame({"portfolio_value": pv, "daily_return": daily_rets}, index=dates)
        return bt

    def test_positive_portfolio_value(self):
        """Patrimônio positivo deve passar."""
        bt = self._make_backtest(np.linspace(10000, 11000, 10))
        errors, warnings = validate_backtest_consistency({"teste": bt}, {"teste": pd.DataFrame()}, pd.DataFrame())
        assert len(errors) == 0

    def test_negative_portfolio_value(self):
        """Patrimônio negativo deve gerar erro."""
        bt = self._make_backtest([10000, 5000, -100, 2000, 3000])
        errors, warnings = validate_backtest_consistency({"teste": bt}, {"teste": pd.DataFrame()}, pd.DataFrame())
        assert any("Patrimônio <= 0" in e for e in errors)


class TestValidateMetrics:
    """Testes para validação de métricas."""

    def test_valid_metrics(self):
        """Métricas válidas devem passar."""
        metrics = pd.DataFrame(
            {
                "total_return": [0.5],
                "annualized_return": [0.2],
                "annualized_volatility": [0.3],
                "sharpe_ratio": [1.0],
                "max_drawdown": [-0.2],
                "calmar_ratio": [1.0],
            },
            index=["teste"],
        )
        errors, warnings = validate_metrics(metrics)
        assert len(errors) == 0

    def test_zero_volatility(self):
        """Volatilidade zero deve gerar erro."""
        metrics = pd.DataFrame(
            {
                "total_return": [0.5],
                "annualized_return": [0.2],
                "annualized_volatility": [0.0],
                "sharpe_ratio": [1.0],
                "max_drawdown": [-0.2],
                "calmar_ratio": [1.0],
            },
            index=["teste"],
        )
        errors, warnings = validate_metrics(metrics)
        assert any("Volatilidade" in e for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
