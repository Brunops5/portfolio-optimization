"""
Testes unitários para o módulo data_loader.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import compute_returns


class TestComputeReturns:
    """Testes para o cálculo de retornos logarítmicos."""

    def test_log_returns_formula(self):
        """Retornos diários devem seguir ln(P_t / P_{t-1})."""
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="B")
        n = len(dates)
        prices = pd.DataFrame({"A": [100, 110, 121, 133.1][:n]}, index=dates)
        result = compute_returns(prices)
        daily = result["daily"]

        # Retorno esperado: ln(110/100) ≈ 0.0953
        assert daily["A"].iloc[0] == pytest.approx(np.log(1.1), abs=1e-6)

    def test_monthly_returns_last_day(self):
        """Retornos mensais devem usar o último dia útil de cada mês."""
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
        # Cresce 10% a cada mês
        values = np.zeros(len(dates))
        for i, d in enumerate(dates):
            month_factor = 1.1 ** (d.month - 1)
            values[i] = 100 * month_factor * (1 + 0.001 * i)
        prices = pd.DataFrame({"A": values}, index=dates)
        result = compute_returns(prices)
        monthly = result["monthly"]

        assert len(monthly) == 2
        assert np.all(np.isfinite(monthly))

    def test_forward_fill_missing_values(self):
        """Valores faltantes devem ser preenchidos via ffill/bfill."""
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="B")
        n = len(dates)
        values = [100, np.nan, 110, np.nan, 121][:n]
        prices = pd.DataFrame({"A": values}, index=dates)
        result = compute_returns(prices)
        assert not result["prices_filled"].isna().any().any()

    def test_output_keys(self):
        """Saída deve conter as chaves esperadas."""
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="B")
        n = len(dates)
        prices = pd.DataFrame({"A": [100, 105, 110, 115, 120][:n]}, index=dates)
        result = compute_returns(prices)
        assert set(result.keys()) == {"prices_filled", "daily", "monthly"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
