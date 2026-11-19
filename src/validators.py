'''Validate mathematical and financial consistency throughout the experiment.'''

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import names
from optimizer import constituent_record_for_date, load_constituent_history

CONFIG_PATH = Path(__file__).resolve().parents[1] / names.CONFIG_FILENAME
with CONFIG_PATH.open('r', encoding='utf-8') as config_file:
    CONFIG = yaml.safe_load(config_file)


def validate_weights(
    weights_dict: Dict[str, pd.DataFrame],
) -> Tuple[List[str], List[str]]:
    '''Check allocation, finiteness, bounds, and non-empty portfolios.'''
    errors = []
    min_allowed = CONFIG[names.OPTIMIZATION][names.MIN_WEIGHT]
    max_allowed = CONFIG[names.OPTIMIZATION][names.MAX_WEIGHT]

    for name, weights in weights_dict.items():
        row_sums = weights.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            errors.append(f'[{name}] Portfolio weights do not sum to 1.')
        if not np.isfinite(weights.to_numpy()).all():
            errors.append(f'[{name}] Non-finite weight detected.')
        if (weights < min_allowed - 1e-7).any().any():
            errors.append(f'[{name}] Weight below the minimum bound detected.')
        if (weights > max_allowed + 1e-7).any().any():
            errors.append(
                f'[{name}] Weight above the maximum bound '
                f'of {max_allowed:.0%} detected.'
            )
        if ((weights > 1e-6).sum(axis=1) == 0).any():
            errors.append(f'[{name}] Empty portfolio detected on at least one date.')

    return errors, []


def validate_historical_constituents(
    weights_dict: Dict[str, pd.DataFrame],
) -> Tuple[List[str], List[str]]:
    '''Require zero weight in assets outside the NCI on each rebalance date.'''
    history = load_constituent_history()
    errors = []

    for strategy, weights in weights_dict.items():
        for date, row in weights.iterrows():
            record = constituent_record_for_date(history, pd.Timestamp(date))
            active = set(record[names.CONSTITUENTS])
            inactive = row.index.difference(list(active))
            invalid = row.loc[inactive].abs() > 1e-7
            if invalid.any():
                assets = sorted(row.loc[inactive].index[invalid].tolist())
                errors.append(
                    f'[{strategy}] {pd.Timestamp(date).date()}: weight outside '
                    f'the effective NCI composition: {assets}.'
                )

    return errors, []


def validate_risk_targets(
    diagnostics: Optional[pd.DataFrame],
) -> Tuple[List[str], List[str]]:
    '''Check covariance, optimizer status, and volatility target compliance.'''
    if diagnostics is None or diagnostics.empty:
        return [], ['Optimization diagnostics were not provided.']

    errors = []
    warnings = []
    if not diagnostics[names.COVARIANCE_PSD].fillna(False).all():
        errors.append('Non-PSD covariance detected after correction.')
    if names.SUCCESS in diagnostics:
        if not diagnostics[names.SUCCESS].fillna(False).all():
            errors.append('An optimization failure was recorded.')

    feasible = diagnostics[names.TARGET_FEASIBLE].fillna(False)
    excess_volatility = (
        diagnostics[names.ANNUAL_VOLATILITY]
        > diagnostics[names.TARGET_VOLATILITY] + 1e-6
    )
    violations = diagnostics.loc[feasible & excess_volatility]
    for _, row in violations.iterrows():
        errors.append(
            f'[{row[names.STRATEGY]}] '
            f'{pd.Timestamp(row[names.REBALANCE_DATE]).date()}: '
            'volatility exceeds the declared target.'
        )

    infeasible = diagnostics.loc[~feasible]
    for _, row in infeasible.iterrows():
        warnings.append(
            f'[{row[names.STRATEGY]}] '
            f'{pd.Timestamp(row[names.REBALANCE_DATE]).date()}: '
            f'target={row[names.TARGET_VOLATILITY]:.3f} is infeasible; '
            f'minimum={row[names.MINIMUM_VOLATILITY]:.3f}.'
        )

    return errors, warnings


def validate_backtest_consistency(
    backtest_results: Dict[str, pd.DataFrame],
    weights_dict: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    '''Check dates, values, returns, drawdowns, and missing-price events.'''
    errors = []
    warnings = []

    for name, backtest in backtest_results.items():
        if name not in weights_dict:
            errors.append(f'[{name}] Backtest has no matching weight table.')
        if not backtest.index.is_unique:
            errors.append(f'[{name}] Backtest dates are not unique.')
        if not backtest.index.is_monotonic_increasing:
            errors.append(f'[{name}] Backtest dates are not sorted.')
        if not backtest.index.isin(daily_returns.index).all():
            errors.append(f'[{name}] Backtest contains unknown return dates.')

        values = backtest[names.PORTFOLIO_VALUE]
        returns = backtest[names.DAILY_RETURN]
        if (values <= 0).any():
            errors.append(f'[{name}] Portfolio value is zero or negative.')
        if returns.max() > 2.0 or returns.min() <= -1.0:
            errors.append(f'[{name}] Daily return is outside the plausible range.')

        implied = values.pct_change(fill_method=None).dropna()
        recorded = returns.reindex(implied.index).dropna()
        difference = (implied.reindex(recorded.index) - recorded).abs().max()
        if pd.notna(difference) and difference > 1e-10:
            errors.append(
                f'[{name}] Return and portfolio value are inconsistent: '
                f'difference={difference:.12f}'
            )

        drawdown = ((values - values.cummax()) / values.cummax()).min()
        if drawdown < -1.0:
            errors.append(f'[{name}] Drawdown is below -100%.')

        if names.STALE_POSITIONS in backtest:
            stale_count = int(backtest[names.STALE_POSITIONS].sum())
            if stale_count > 0:
                message = (
                    f'[{name}] {stale_count} holding-days without a new price '
                    'were kept at their last value.'
                )
                stale_is_error = CONFIG[names.BACKTEST].get(
                    names.STALE_POSITIONS_ARE_ERRORS,
                    True,
                )
                if stale_is_error:
                    errors.append(message)
                else:
                    warnings.append(message)

    missing_backtests = set(weights_dict) - set(backtest_results)
    for name in sorted(missing_backtests):
        errors.append(f'[{name}] Weight table has no matching backtest.')

    return errors, warnings


def validate_metrics(metrics_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    '''Apply basic sanity checks to the final performance metrics.'''
    errors = []
    for strategy, row in metrics_df.iterrows():
        if not -1.0 <= row[names.TOTAL_RETURN] <= 10.0:
            errors.append(f'[{strategy}] Suspicious total return.')
        if row[names.ANNUALIZED_VOLATILITY] < 1e-6:
            errors.append(f'[{strategy}] Annualized volatility is near zero.')
    return errors, []


def run_all_validations(
    weights_dict: Dict[str, pd.DataFrame],
    backtest_results: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
    metrics_df: pd.DataFrame,
    profiles: Dict,
    optimization_diagnostics: Optional[pd.DataFrame] = None,
) -> bool:
    '''Run every validation and return whether the experiment can be published.'''
    expected_strategies = {*profiles, names.BENCHMARK}
    actual_strategies = set(weights_dict)
    strategy_errors = []
    if actual_strategies != expected_strategies:
        strategy_errors.append(
            'Generated strategies differ from the configuration: '
            f'expected={sorted(expected_strategies)}, '
            f'actual={sorted(actual_strategies)}.'
        )

    checks = [
        ('Strategies', strategy_errors, []),
        ('Weights', *validate_weights(weights_dict)),
        (
            'Historical constituents',
            *validate_historical_constituents(weights_dict),
        ),
        (
            'Risk targets and PSD',
            *validate_risk_targets(optimization_diagnostics),
        ),
        (
            'Backtest',
            *validate_backtest_consistency(
                backtest_results,
                weights_dict,
                daily_returns,
            ),
        ),
        ('Metrics', *validate_metrics(metrics_df)),
    ]

    all_errors = []
    warning_count = 0
    print('\n' + '=' * 60)
    print('EXPERIMENT VALIDATION')
    print('=' * 60)

    for name, errors, warnings in checks:
        print(f'\n[{name}]')
        for error in errors[:5]:
            print(f'  ERROR: {error}')
        for warning in warnings[:5]:
            print(f'  WARNING: {warning}')
        if not errors and not warnings:
            print('  Passed.')
        all_errors.extend(errors)
        warning_count += len(warnings)

    print(f'\nErrors: {len(all_errors)} | Warnings: {warning_count}')
    return not all_errors
