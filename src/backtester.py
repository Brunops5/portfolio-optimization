'''Simulate portfolios that buy and hold between scheduled rebalances.'''

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import names

CONFIG_PATH = Path(__file__).resolve().parents[1] / names.CONFIG_FILENAME
with CONFIG_PATH.open('r', encoding='utf-8') as config_file:
    CONFIG = yaml.safe_load(config_file)


def _validated_target_weights(
    row: pd.Series,
    columns: pd.Index,
) -> np.ndarray:
    '''Return target weights aligned with the return columns.

    Missing assets receive zero weight. Invalid weights are rejected instead of
    being silently repaired, because that would change the experiment.
    '''
    target = row.reindex(columns).fillna(0.0).to_numpy(dtype=float)

    if not np.all(np.isfinite(target)):
        raise ValueError('Rebalance weights must be finite.')
    if (target < -1e-8).any():
        raise ValueError('Rebalance weights cannot be negative.')

    target = np.maximum(target, 0.0)
    if target.sum() <= 0:
        raise ValueError('The portfolio must contain at least one positive weight.')
    if not np.isclose(target.sum(), 1.0, atol=1e-6):
        raise ValueError('Rebalance weights must sum to 1.')

    return target / target.sum()


def backtest_strategy(
    daily_returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    '''Simulate daily holdings, weekly rebalances, and transaction costs.

    Each day follows three steps:
    1. Rebalance when the date has a new target weight vector.
    2. Deduct cost once, using turnover times the transaction cost rate.
    3. Apply the simple return of each asset to its monetary holding.

    Asset weights therefore drift between rebalances. A missing return keeps the
    last monetary value of that holding and is reported in ``stale_positions``.
    '''
    if weights.empty:
        raise ValueError('The weight table is empty.')
    if not 0 <= transaction_cost < 1:
        raise ValueError('transaction_cost must be in the interval [0, 1).')

    returns = daily_returns.sort_index()
    returns = returns.loc[returns.index >= weights.index[0]].copy()
    if returns.empty:
        raise ValueError('There are no returns after the first rebalance.')

    target_by_date = weights.sort_index()
    first_target = _validated_target_weights(
        target_by_date.iloc[0],
        returns.columns,
    )
    holdings = initial_capital * first_target
    portfolio_value = float(initial_capital)
    records = []

    for date, row in returns.iterrows():
        value_before = portfolio_value
        transaction_cost_value = 0.0
        turnover = 0.0
        asset_returns = row.to_numpy(dtype=float)
        missing_returns = np.isnan(asset_returns)
        held_before_rebalance = holdings > 1e-10

        is_rebalance = date in target_by_date.index
        is_first_date = date == target_by_date.index[0]
        if is_rebalance and not is_first_date:
            current_weights = holdings / holdings.sum()
            target = _validated_target_weights(
                target_by_date.loc[date],
                returns.columns,
            )
            turnover = float(np.abs(target - current_weights).sum())
            transaction_cost_value = portfolio_value * transaction_cost * turnover
            portfolio_value -= transaction_cost_value
            holdings = portfolio_value * target

        held_after_rebalance = holdings > 1e-10
        held_assets = held_before_rebalance | held_after_rebalance
        missing_held = held_assets & missing_returns

        # Missing prices preserve holdings, but the validator can reject them.
        asset_returns = np.nan_to_num(asset_returns, nan=0.0)
        if (asset_returns <= -1).any():
            raise ValueError(
                f'Simple return less than or equal to -100% on {date.date()}.'
            )

        holdings *= 1.0 + asset_returns
        portfolio_value = float(holdings.sum())
        net_return = portfolio_value / value_before - 1.0
        records.append(
            {
                names.PORTFOLIO_VALUE: portfolio_value,
                names.DAILY_RETURN: net_return,
                names.TURNOVER: turnover,
                names.TRANSACTION_COST: transaction_cost_value,
                names.STALE_POSITIONS: int(missing_held.sum()),
            }
        )

    return pd.DataFrame(records, index=returns.index)


def compute_drawdown_series(
    portfolio_value: pd.Series,
    initial_capital: Optional[float] = None,
) -> pd.Series:
    '''Calculate the percentage distance from each historical value peak.'''
    rolling_max = portfolio_value.cummax()
    if initial_capital is not None:
        rolling_max = rolling_max.clip(lower=float(initial_capital))
    return (portfolio_value - rolling_max) / rolling_max


def maximum_drawdown(
    portfolio_value: pd.Series,
    initial_capital: Optional[float] = None,
) -> float:
    '''Return the largest peak-to-trough loss in the series.'''
    drawdown = compute_drawdown_series(portfolio_value, initial_capital)
    return float(drawdown.min())


def calculate_metrics(
    portfolio_value: pd.Series,
    daily_returns: pd.Series,
    initial_capital: Optional[float] = None,
) -> Dict:
    '''Calculate total return, annualized return, volatility, and drawdown.'''
    if portfolio_value.empty or daily_returns.empty:
        raise ValueError('Metrics cannot be calculated from empty series.')
    if initial_capital is None:
        first_return = daily_returns.iloc[0]
        initial_capital = float(portfolio_value.iloc[0] / (1 + first_return))

    total_return = float(portfolio_value.iloc[-1] / initial_capital - 1)
    if isinstance(portfolio_value.index, pd.DatetimeIndex):
        elapsed_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days + 1
        years = elapsed_days / names.ANNUALIZATION_FACTOR
    else:
        years = len(daily_returns) / names.ANNUALIZATION_FACTOR

    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0.0

    daily_std = daily_returns.std(ddof=1)
    annualized_volatility = float(daily_std * np.sqrt(names.ANNUALIZATION_FACTOR))
    max_drawdown = maximum_drawdown(portfolio_value, initial_capital)

    return {
        names.TOTAL_RETURN: total_return,
        names.ANNUALIZED_RETURN: float(annualized_return),
        names.ANNUALIZED_VOLATILITY: annualized_volatility,
        names.MAX_DRAWDOWN: max_drawdown,
    }


def run_full_backtest(
    daily_returns: pd.DataFrame,
    weights_dict: Dict[str, pd.DataFrame],
    transaction_cost: Optional[float] = None,
    initial_capital: Optional[float] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    '''Backtest every strategy and collect comparable performance metrics.'''
    if transaction_cost is None:
        transaction_cost = CONFIG[names.BACKTEST][names.TRANSACTION_COST]
    if initial_capital is None:
        initial_capital = CONFIG[names.BACKTEST][names.INITIAL_CAPITAL]

    print('\n' + '=' * 60)
    print('STEP 3 — BACKTESTING')
    print('=' * 60)

    results = {}
    metric_rows = []
    for name, strategy_weights in weights_dict.items():
        result = backtest_strategy(
            daily_returns,
            strategy_weights,
            transaction_cost,
            initial_capital,
        )
        metrics = calculate_metrics(
            result[names.PORTFOLIO_VALUE],
            result[names.DAILY_RETURN],
            initial_capital,
        )

        elapsed_days = (result.index[-1] - result.index[0]).days + 1
        elapsed_years = elapsed_days / names.ANNUALIZATION_FACTOR
        total_turnover = float(result[names.TURNOVER].sum())
        if elapsed_years > 0:
            annualized_turnover = total_turnover / elapsed_years
        else:
            annualized_turnover = 0.0

        metrics[names.TOTAL_TURNOVER] = total_turnover
        metrics[names.ANNUALIZED_TURNOVER] = annualized_turnover
        metrics[names.TOTAL_TRANSACTION_COST] = float(
            result[names.TRANSACTION_COST].sum()
        )
        metrics[names.REBALANCE_EVENTS] = int((result[names.TURNOVER] > 1e-12).sum())
        metrics[names.STRATEGY] = name
        results[name] = result
        metric_rows.append(metrics)

        print(
            f'{name}: return={metrics[names.TOTAL_RETURN]:.2%}, '
            f'volatility={metrics[names.ANNUALIZED_VOLATILITY]:.2%}, '
            f'max drawdown={metrics[names.MAX_DRAWDOWN]:.2%}'
        )

    return results, pd.DataFrame(metric_rows).set_index(names.STRATEGY)


def save_backtest_results(
    results: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    outdir: Path,
) -> None:
    '''Save one daily backtest per strategy and a shared metrics table.'''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, dataframe in results.items():
        dataframe.to_csv(outdir / f'backtest_{name}.csv')

    metrics_path = outdir / names.METRICS_SUMMARY_FILENAME
    metrics_df.to_csv(metrics_path)
    print(f'[SAVED] Metrics: {metrics_path}')
