"""Tests for line-only figures and efficient frontier point labels."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import names
import visualization


@pytest.fixture
def captured_figures(monkeypatch):
    figures = []
    monkeypatch.setattr(
        visualization,
        "_save_figure",
        lambda figure, save_path: figures.append(figure),
    )
    yield figures
    plt.close("all")


def _backtest_results():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return {
        names.RISK_065: pd.DataFrame(
            {
                "portfolio_value": np.linspace(10000, 11000, len(dates)),
            },
            index=dates,
        ),
        names.BENCHMARK: pd.DataFrame(
            {
                "portfolio_value": np.linspace(10000, 10500, len(dates)),
            },
            index=dates,
        ),
    }


def test_portfolio_evolution_uses_lines_only(captured_figures):
    visualization.plot_portfolio_evolution(_backtest_results())

    axes = captured_figures[0].axes[0]
    assert len(axes.lines) == 2
    assert not axes.collections


def test_drawdowns_use_lines_only(captured_figures):
    visualization.plot_drawdowns(_backtest_results())

    axes = captured_figures[0].axes[0]
    assert len(axes.lines) == 3
    assert not axes.collections


def test_efficient_frontier_combines_identical_points(
    monkeypatch,
    captured_figures,
):
    generator = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    returns = pd.DataFrame(
        generator.normal(
            [0.0003, 0.0005, 0.0007],
            [0.015, 0.025, 0.035],
            (len(dates), 3),
        ),
        index=dates,
        columns=["BTC", "ETH", "SOL"],
    )
    monkeypatch.setattr(
        visualization,
        "load_constituent_history",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        visualization,
        "prepare_estimation_window",
        lambda *args: (returns, None, list(returns.columns)),
    )

    rebalance_date = dates[-1] + pd.Timedelta(days=1)
    identical_weights = pd.DataFrame(
        {"BTC": [1.0], "ETH": [0.0], "SOL": [0.0]},
        index=[rebalance_date],
    )
    weights = {
        names.RISK_085: identical_weights,
        names.RISK_105: identical_weights.copy(),
    }

    visualization.plot_efficient_frontier(
        returns,
        weights,
        n_random=100,
    )

    axes = captured_figures[0].axes[0]
    legend_labels = [text.get_text() for text in axes.legend_.get_texts()]
    assert "Moderate (85%) / Aggressive (105%)" in legend_labels
    assert not axes.texts
