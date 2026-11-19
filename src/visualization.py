'''Create publication-ready figures for the portfolio experiment.'''

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from pypfopt import EfficientFrontier

import names
from backtester import compute_drawdown_series
from optimizer import (
    estimate_inputs,
    load_constituent_history,
    prepare_estimation_window,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / names.CONFIG_FILENAME
with CONFIG_PATH.open('r', encoding='utf-8') as config_file:
    CONFIG = yaml.safe_load(config_file)

PALETTE = {
    names.RISK_065: '#1B5E20',
    names.RISK_085: '#1565C0',
    names.RISK_105: '#C62828',
    names.BENCHMARK: '#616161',
}

PUBLICATION_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.color': '#BBBBBB',
    'grid.linestyle': '--',
}


def _strategy_label(strategy: str) -> str:
    if strategy == names.BENCHMARK:
        return 'Benchmark 1/N'
    profile = CONFIG[names.PROFILES].get(strategy)
    if profile is not None:
        return profile[names.PROFILE_NAME]
    return strategy.replace('_', ' ').title()


def _save_figure(fig: plt.Figure, save_path: Optional[Path]) -> None:
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'[SAVED] {save_path}')
    plt.close(fig)


def plot_portfolio_evolution(
    backtest_results: Dict[str, pd.DataFrame],
    log_scale: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    '''Plot portfolio values as simple lines without filled areas.'''
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, dataframe in backtest_results.items():
        ax.plot(
            dataframe.index,
            dataframe[names.PORTFOLIO_VALUE],
            label=_strategy_label(name),
            linewidth=2.0,
            color=PALETTE.get(name, '#333333'),
        )

    ax.set_title('Net Portfolio Value Over Time', loc='left', pad=15)
    ax.set_xlabel('Date', labelpad=10)
    ax.set_ylabel('Portfolio value (US$)', labelpad=10)
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Portfolio value (US$, log scale)', labelpad=10)

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    ax.legend(loc='upper left', frameon=True)
    fig.autofmt_xdate(rotation=30)
    sns.despine(ax=ax, top=True, right=True)
    _save_figure(fig, save_path)


def plot_drawdowns(
    backtest_results: Dict[str, pd.DataFrame],
    initial_capital: float = 10000.0,
    save_path: Optional[Path] = None,
) -> None:
    '''Plot drawdowns as independent lines without overlapping filled areas.'''
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, dataframe in backtest_results.items():
        drawdown = compute_drawdown_series(
            dataframe[names.PORTFOLIO_VALUE],
            initial_capital=initial_capital,
        )
        ax.plot(
            drawdown.index,
            drawdown,
            label=_strategy_label(name),
            linewidth=1.6,
            color=PALETTE.get(name, '#333333'),
        )

    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_title('Drawdown Over Time', loc='left', pad=15)
    ax.set_xlabel('Date', labelpad=10)
    ax.set_ylabel('Drawdown', labelpad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    ax.legend(loc='lower left', frameon=True)
    fig.autofmt_xdate(rotation=30)
    sns.despine(ax=ax, top=True, right=True)
    _save_figure(fig, save_path)


def plot_allocation_heatmap(
    weights_df: pd.DataFrame,
    title: str = 'Weekly Allocation',
    save_path: Optional[Path] = None,
) -> None:
    '''Show asset weights at each rebalance date as a heatmap.'''
    label_step = max(1, int(np.ceil(len(weights_df.index) / 24)))
    tick_labels = [
        date.strftime('%Y-%m-%d') if position % label_step == 0 else ''
        for position, date in enumerate(weights_df.index)
    ]

    number_of_dates = len(weights_df.index)
    number_of_assets = len(weights_df.columns)
    fig_width = min(24, max(12, number_of_dates * 0.15))
    fig_height = max(5, number_of_assets * 0.6)
    show_values = number_of_dates <= 24 and number_of_assets <= 12

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        weights_df.T * 100,
        cmap='Blues',
        annot=show_values,
        fmt='.1f',
        annot_kws={'size': 10, 'color': '#111111'},
        cbar_kws={'label': 'Weight (%)', 'shrink': 0.6, 'pad': 0.02},
        linewidths=0.3,
        linecolor='#E0E0E0',
        ax=ax,
        xticklabels=tick_labels,
        yticklabels=True,
    )

    ax.set_title(title, loc='left', pad=15, fontweight='bold')
    ax.set_xlabel('Rebalance date', labelpad=10)
    ax.set_ylabel('Asset', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    _save_figure(fig, save_path)


def _efficient_frontier_points(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    bounds: tuple[float, float],
) -> tuple[list[float], list[float], float, float, float]:
    '''Calculate the efficient curve from minimum risk to maximum return.'''
    minimum_frontier = EfficientFrontier(
        expected_returns,
        covariance,
        weight_bounds=bounds,
    )
    minimum_frontier.min_volatility()
    minimum_return, minimum_volatility, _ = minimum_frontier.portfolio_performance()

    # No long-only portfolio can exceed the largest individual asset volatility.
    maximum_asset_volatility = float(np.sqrt(np.diag(covariance)).max())
    maximum_frontier = EfficientFrontier(
        expected_returns,
        covariance,
        weight_bounds=bounds,
    )
    maximum_frontier.efficient_risk(maximum_asset_volatility * 1.001)
    maximum_return_volatility = maximum_frontier.portfolio_performance()[1]

    frontier_returns = []
    frontier_volatilities = []
    if maximum_return_volatility > minimum_volatility + 1e-8:
        targets = np.linspace(
            minimum_volatility * (1 + 1e-5),
            maximum_return_volatility * (1 + 1e-5),
            80,
        )
        for target in targets:
            frontier = EfficientFrontier(
                expected_returns,
                covariance,
                weight_bounds=bounds,
            )
            frontier.efficient_risk(float(target))
            annual_return, annual_volatility, _ = frontier.portfolio_performance()
            frontier_returns.append(annual_return)
            frontier_volatilities.append(annual_volatility)

    return (
        frontier_volatilities,
        frontier_returns,
        minimum_volatility,
        minimum_return,
        maximum_return_volatility,
    )


def plot_efficient_frontier(
    daily_returns: pd.DataFrame,
    weights_dict: Dict[str, pd.DataFrame],
    n_random: int = 5000,
    save_path: Optional[Path] = None,
) -> None:
    '''Plot the feasible set and exact efficient frontier for the last window.

    The curve is obtained by repeatedly solving the same risk-constrained
    optimization used by the experiment. Simulated portfolios provide context;
    they are not used to approximate the frontier itself.
    '''
    last_rebalance = max(weights.index.max() for weights in weights_dict.values())
    constituent_history = load_constituent_history()
    estimation_window, _, eligible = prepare_estimation_window(
        daily_returns,
        last_rebalance,
        constituent_history,
    )
    if estimation_window.shape[1] < 2:
        raise ValueError('Insufficient data to plot the efficient frontier.')

    expected_returns, covariance, _ = estimate_inputs(estimation_window)
    mean_returns = expected_returns.to_numpy()
    cov_matrix = covariance.to_numpy()
    number_of_assets = len(mean_returns)

    generator = np.random.default_rng(42)
    candidates = generator.dirichlet(
        np.ones(number_of_assets),
        n_random * 4,
    )
    min_weight = CONFIG[names.OPTIMIZATION][names.MIN_WEIGHT]
    max_weight = CONFIG[names.OPTIMIZATION][names.MAX_WEIGHT]
    within_bounds = (candidates.min(axis=1) >= min_weight - 1e-12) & (
        candidates.max(axis=1) <= max_weight + 1e-12
    )
    valid = candidates[within_bounds]
    if len(valid) < n_random:
        raise ValueError('Could not simulate enough portfolios within the bounds.')

    random_weights = valid[:n_random]
    random_returns = random_weights @ mean_returns
    random_variances = np.einsum(
        'ij,jk,ik->i',
        random_weights,
        cov_matrix,
        random_weights,
    )
    random_volatilities = np.sqrt(random_variances)

    bounds = (min_weight, max_weight)
    (
        frontier_volatilities,
        frontier_returns,
        minimum_volatility,
        minimum_return,
        _,
    ) = _efficient_frontier_points(expected_returns, covariance, bounds)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(
        random_volatilities,
        random_returns,
        color='#4C78A8',
        s=8,
        alpha=0.2,
        edgecolors='none',
        label='Simulated feasible portfolios',
    )

    if frontier_volatilities:
        ax.plot(
            frontier_volatilities,
            frontier_returns,
            color='#111111',
            linewidth=2.5,
            label='Efficient frontier',
            zorder=4,
        )
    else:
        ax.scatter(
            minimum_volatility,
            minimum_return,
            facecolors='none',
            edgecolors='#111111',
            linewidths=1.5,
            marker='*',
            s=260,
            label='Efficient frontier (single point)',
            zorder=7,
        )

    # Vertical lines show risk ceilings; they are not realized risk estimates.
    for name, profile in CONFIG[names.PROFILES].items():
        ax.axvline(
            profile[names.TARGET_VOLATILITY],
            color=PALETTE.get(name, '#333333'),
            linestyle='--',
            linewidth=1.0,
            alpha=0.55,
        )

    portfolio_points = []
    for name, strategy_weights in weights_dict.items():
        weights = strategy_weights.loc[last_rebalance]
        weights = weights.reindex(eligible).fillna(0).to_numpy(dtype=float)
        if weights.sum() <= 0:
            continue

        weights = weights / weights.sum()
        annual_return = float(weights @ mean_returns)
        annual_volatility = float(np.sqrt(weights @ cov_matrix @ weights))
        portfolio_points.append(
            (annual_volatility, annual_return, _strategy_label(name), name)
        )

    # Combine identical points instead of moving them away from their true values.
    grouped_points = {}
    for volatility, annual_return, label, name in portfolio_points:
        coordinate = (round(volatility, 6), round(annual_return, 6))
        grouped_points.setdefault(coordinate, []).append((label, name))

    for (volatility, annual_return), entries in grouped_points.items():
        labels = [entry[0] for entry in entries]
        strategy_names = [entry[1] for entry in entries]
        color = PALETTE.get(strategy_names[0], '#333333')
        marker = 'o' if names.BENCHMARK in strategy_names else 'D'
        ax.scatter(
            volatility,
            annual_return,
            color=color,
            s=130,
            zorder=5,
            edgecolors='white',
            linewidths=1.5,
            marker=marker,
            label=' / '.join(labels),
        )

    ax.set_title(
        'Efficient Frontier and Feasible Portfolios — Last Estimation Window',
        loc='left',
        pad=15,
    )
    ax.set_xlabel('Annualized volatility', labelpad=10)
    ax.set_ylabel('Expected annualized return', labelpad=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc='best', frameon=True)
    sns.despine(ax=ax, top=True, right=True)
    _save_figure(fig, save_path)


def generate_all_figures(
    backtest_results: Dict[str, pd.DataFrame],
    weights_dict: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
    outdir: Path,
) -> None:
    '''Generate every figure from validated experiment outputs.'''
    print('\n' + '=' * 60)
    print('STEP 5 — VISUALIZATION')
    print('=' * 60)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PUBLICATION_STYLE):
        plot_portfolio_evolution(
            backtest_results,
            save_path=outdir / names.PORTFOLIO_EVOLUTION_FILENAME,
        )
        plot_drawdowns(
            backtest_results,
            initial_capital=CONFIG[names.BACKTEST][names.INITIAL_CAPITAL],
            save_path=outdir / names.DRAWDOWNS_FILENAME,
        )
        plot_efficient_frontier(
            daily_returns,
            weights_dict,
            save_path=outdir / names.EFFICIENT_FRONTIER_FILENAME,
        )
        for name, weights in weights_dict.items():
            plot_allocation_heatmap(
                weights,
                title=f'Weekly Allocation — {_strategy_label(name)}',
                save_path=outdir / f'heatmap_{name}.pdf',
            )

    print(f'\n[ALL FIGURES SAVED IN] {outdir}')
