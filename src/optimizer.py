"""Estimate and optimize Markowitz portfolios in rolling time windows."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pypfopt import EfficientFrontier, expected_returns, risk_models

import names

CONFIG_PATH = Path(__file__).resolve().parents[1] / names.CONFIG_FILENAME
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


def load_constituent_history(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the point-in-time NCI constituent history.

    Each row contains the date on which a quarterly composition became active.
    Using this effective date prevents future constituents from entering an
    earlier optimization window.
    """
    if path is None:
        path = CONFIG_PATH.parent / CONFIG[names.PATHS][names.CONSTITUENT_HISTORY]

    with Path(path).open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)

    if isinstance(document, dict):
        rows = document.get(names.SCHEDULE, [])
    else:
        rows = []
    if not rows:
        raise ValueError("The NCI constituent history is empty.")

    history = pd.DataFrame(rows)
    required_columns = {
        names.EFFECTIVE_DATE,
        names.QUARTER,
        names.CONSTITUENTS,
    }
    if not required_columns.issubset(history.columns):
        raise ValueError("The NCI history is missing required fields.")

    history[names.EFFECTIVE_DATE] = pd.to_datetime(history[names.EFFECTIVE_DATE])
    if history[names.EFFECTIVE_DATE].duplicated().any():
        raise ValueError("The NCI history contains duplicate effective dates.")

    history = history.sort_values(names.EFFECTIVE_DATE)
    history = history.set_index(names.EFFECTIVE_DATE)
    history[names.CONSTITUENTS] = history[names.CONSTITUENTS].map(tuple)

    historical_assets = {
        asset for constituents in history[names.CONSTITUENTS] for asset in constituents
    }
    missing_assets = historical_assets - set(CONFIG[names.ASSETS])
    if missing_assets:
        raise ValueError(
            "Historical constituents missing from config.assets: "
            f"{sorted(missing_assets)}"
        )

    return history


def constituent_record_for_date(
    history: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.Series:
    """Return the latest NCI composition effective on the given date."""
    available = history.loc[history.index <= pd.Timestamp(date)]
    if available.empty:
        raise ValueError(f"No NCI composition was effective on {date}.")
    return available.iloc[-1]


def estimate_inputs(
    simple_returns: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """Estimate annualized expected returns and covariance.

    PyPortfolioOpt calculates the historical mean and sample covariance. The
    annualization factor is 365 because cryptocurrency markets trade every day.
    Spectral correction removes small negative eigenvalues caused by numerical
    noise. Both pre- and post-correction eigenvalues are kept for validation.
    """
    if simple_returns.empty or simple_returns.shape[1] < 2:
        raise ValueError("Returns from at least two assets are required.")
    if simple_returns.isna().any().any():
        raise ValueError("The estimation window contains missing values.")

    mu = expected_returns.mean_historical_return(
        simple_returns,
        returns_data=True,
        frequency=names.ANNUALIZATION_FACTOR,
    )

    # Calculate the raw matrix separately to make the PSD diagnostic explicit.
    raw_covariance = simple_returns.cov() * names.ANNUALIZATION_FACTOR
    min_eigenvalue_before = float(np.linalg.eigvalsh(raw_covariance.values).min())
    covariance = risk_models.sample_cov(
        simple_returns,
        returns_data=True,
        frequency=names.ANNUALIZATION_FACTOR,
        fix_method=names.SPECTRAL,
    )
    min_eigenvalue_after = float(np.linalg.eigvalsh(covariance.values).min())

    diagnostics = {
        names.MIN_EIGENVALUE_BEFORE: min_eigenvalue_before,
        names.MIN_EIGENVALUE_AFTER: min_eigenvalue_after,
        names.COVARIANCE_PSD: min_eigenvalue_after >= -1e-10,
    }
    return mu, covariance, diagnostics


def optimize_portfolio(
    simple_returns: pd.DataFrame,
    target_volatility: float,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """Maximize expected return without exceeding annualized volatility.

    The constraints are full investment and the configured weight bounds. If
    the target is below the minimum feasible volatility, the function returns
    the minimum-volatility portfolio and marks ``target_feasible`` as false.
    """
    if target_volatility <= 0:
        raise ValueError("target_volatility must be positive.")
    if not 0 <= min_weight <= max_weight <= 1:
        raise ValueError("Weight bounds must satisfy 0 <= min <= max <= 1.")

    mu, covariance, covariance_info = estimate_inputs(simple_returns)
    bounds = (min_weight, max_weight)

    minimum_frontier = EfficientFrontier(
        mu,
        covariance,
        weight_bounds=bounds,
    )
    minimum_frontier.min_volatility()
    minimum_volatility = float(minimum_frontier.portfolio_performance()[1])

    target_feasible = target_volatility + 1e-7 >= minimum_volatility
    if target_feasible:
        frontier = EfficientFrontier(mu, covariance, weight_bounds=bounds)
        frontier.efficient_risk(target_volatility=float(target_volatility))
        weights = frontier.weights
        message = "ok"
    else:
        weights = minimum_frontier.weights
        message = (
            f"infeasible target ({target_volatility:.4f}); "
            f"minimum volatility={minimum_volatility:.4f}"
        )

    weights = np.asarray(weights, dtype=float)
    weights[np.abs(weights) < 1e-10] = 0.0
    weights /= weights.sum()

    portfolio_return = float(weights @ mu.values)
    portfolio_variance = weights @ covariance.values @ weights
    portfolio_volatility = float(np.sqrt(portfolio_variance))
    info = {
        names.SUCCESS: True,
        names.MESSAGE: message,
        names.OBJECTIVE: "max_return_for_target_volatility",
        names.TARGET_VOLATILITY: float(target_volatility),
        names.TARGET_FEASIBLE: bool(target_feasible),
        names.MINIMUM_VOLATILITY: minimum_volatility,
        names.EXPECTED_ANNUAL_RETURN: portfolio_return,
        names.ANNUAL_VOLATILITY: portfolio_volatility,
        names.NUM_ASSETS_SELECTED: int((weights > 1e-6).sum()),
        names.MAX_WEIGHT_OBSERVED: float(weights.max()),
        **covariance_info,
    }
    return weights, info


def _rebalance_dates(
    daily_returns: pd.DataFrame,
    window_months: int,
    frequency: str,
) -> pd.DatetimeIndex:
    """Select the first observed day of each period after the initial window."""
    observed_dates = pd.Series(
        daily_returns.index,
        index=daily_returns.index,
    )

    # With W-MON, left-closed bins run from one Monday to the next Monday.
    first_observed_by_period = (
        observed_dates.resample(
            frequency,
            closed="left",
            label="left",
        )
        .min()
        .dropna()
    )
    first_allowed = (
        daily_returns.index.min().to_period("M") + window_months
    ).start_time
    selected = first_observed_by_period.loc[first_observed_by_period >= first_allowed]
    return pd.DatetimeIndex(selected.to_numpy())


def prepare_estimation_window(
    daily_returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    constituent_history: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build a historical window using only eligible NCI constituents.

    Eligibility requires enough observations and a recent last price. Rows with
    any missing eligible asset are removed so every covariance entry uses the
    same dates. The rebalance date itself is excluded to avoid look-ahead.
    """
    constituent_record = constituent_record_for_date(
        constituent_history,
        rebalance_date,
    )
    active_constituents = list(constituent_record[names.CONSTITUENTS])
    missing_columns = set(active_constituents) - set(daily_returns.columns)
    if missing_columns:
        raise ValueError(
            f"NCI constituents without return series on {rebalance_date.date()}: "
            f"{sorted(missing_columns)}"
        )

    window_start = rebalance_date - pd.DateOffset(
        months=CONFIG[names.ESTIMATION_WINDOW_MONTHS]
    )
    in_window = daily_returns.index >= window_start
    before_rebalance = daily_returns.index < rebalance_date
    window = daily_returns.loc[in_window & before_rebalance]

    max_data_gap = CONFIG[names.OPTIMIZATION][names.MAX_DATA_GAP_DAYS]
    recent_cutoff = rebalance_date - pd.Timedelta(days=max_data_gap)
    min_observations = CONFIG[names.OPTIMIZATION][names.MIN_OBSERVATIONS]

    eligible = []
    for asset in active_constituents:
        observations = window[asset].dropna()
        has_enough_history = len(observations) >= min_observations
        has_recent_data = (
            not observations.empty and observations.index.max() >= recent_cutoff
        )
        if has_enough_history and has_recent_data:
            eligible.append(asset)

    estimation_returns = window[eligible].dropna(how="any")
    return estimation_returns, constituent_record, eligible


def run_rolling_optimization(
    daily_returns: pd.DataFrame,
    profiles: Optional[Dict] = None,
    constituent_history: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Optimize every risk profile on each valid rebalance date.

    All profiles and the 1/N benchmark share the same dates, eligible assets,
    and historical estimation window, making their backtests comparable.
    """
    if profiles is None:
        profiles = CONFIG[names.PROFILES]
    if constituent_history is None:
        constituent_history = load_constituent_history()

    optimization_config = CONFIG[names.OPTIMIZATION]
    window_months = CONFIG[names.ESTIMATION_WINDOW_MONTHS]
    rebalance_frequency = CONFIG[names.REBALANCE_FREQUENCY]
    min_observations = optimization_config[names.MIN_OBSERVATIONS]
    all_assets = daily_returns.columns.tolist()

    by_strategy = {name: [] for name in [*profiles, names.BENCHMARK]}
    accepted_dates = []
    diagnostics = []
    rebalance_dates = _rebalance_dates(
        daily_returns,
        window_months,
        rebalance_frequency,
    )

    for rebalance_date in rebalance_dates:
        estimation_returns, constituent_record, eligible = prepare_estimation_window(
            daily_returns,
            rebalance_date,
            constituent_history,
        )
        active_constituents = list(constituent_record[names.CONSTITUENTS])
        window_start = rebalance_date - pd.DateOffset(months=window_months)

        if estimation_returns.shape[1] < 2:
            print(f"[SKIP] Insufficient assets on {rebalance_date.date()}")
            continue
        if len(estimation_returns) < min_observations:
            print(f"[SKIP] Insufficient observations on {rebalance_date.date()}")
            continue

        accepted_dates.append(rebalance_date)
        print(
            f"\n[REBALANCE] {rebalance_date.date()} | "
            f"Window: {window_start.date()} to "
            f"{(rebalance_date - pd.Timedelta(days=1)).date()} | "
            f"Assets: {eligible}"
        )

        for profile_key, profile_config in profiles.items():
            target_volatility = profile_config[names.TARGET_VOLATILITY]
            profile_name = profile_config[names.PROFILE_NAME]
            try:
                weights, info = optimize_portfolio(
                    estimation_returns,
                    target_volatility=target_volatility,
                    min_weight=optimization_config[names.MIN_WEIGHT],
                    max_weight=optimization_config[names.MAX_WEIGHT],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not optimize {profile_name} on "
                    f"{rebalance_date.date()}: {exc}"
                ) from exc

            aligned = pd.Series(0.0, index=all_assets)
            aligned[eligible] = weights
            by_strategy[profile_key].append(aligned.values)
            diagnostics.append(
                {
                    names.REBALANCE_DATE: rebalance_date,
                    names.STRATEGY: profile_key,
                    names.OBSERVATIONS: len(estimation_returns),
                    names.NCI_EFFECTIVE_DATE: constituent_record.name,
                    names.NCI_QUARTER: constituent_record[names.QUARTER],
                    names.NCI_CONSTITUENT_COUNT: len(active_constituents),
                    names.ELIGIBLE_COUNT: len(eligible),
                    names.EXCLUDED_CONSTITUENTS: "|".join(
                        sorted(set(active_constituents) - set(eligible))
                    ),
                    **info,
                }
            )

            if info[names.TARGET_FEASIBLE]:
                status = ""
            else:
                status = " [INFEASIBLE TARGET]"
            print(
                f"  {profile_name:12s}: "
                f"volatility={info[names.ANNUAL_VOLATILITY]:.3f}/"
                f"{info[names.TARGET_VOLATILITY]:.3f}, "
                f"assets={info[names.NUM_ASSETS_SELECTED]}{status}"
            )

        benchmark = pd.Series(0.0, index=all_assets)
        benchmark[eligible] = 1.0 / len(eligible)
        by_strategy[names.BENCHMARK].append(benchmark.values)

    result = {
        name: pd.DataFrame(values, index=accepted_dates, columns=all_assets)
        for name, values in by_strategy.items()
        if values
    }
    return result, pd.DataFrame(diagnostics)


def save_weights(weights_dict: Dict[str, pd.DataFrame], outdir: Path) -> None:
    """Save one CSV containing the rebalance weights of each strategy."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, dataframe in weights_dict.items():
        path = outdir / f"weights_{name}.csv"
        dataframe.to_csv(path)
        print(f"[SAVED] {name} weights: {path}")


def save_optimization_diagnostics(
    diagnostics: pd.DataFrame,
    outdir: Path,
) -> None:
    """Save feasibility, risk, and covariance diagnostics for every window."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / names.OPTIMIZATION_DIAGNOSTICS_FILENAME
    diagnostics.to_csv(path, index=False)
    print(f"[SAVED] Optimization diagnostics: {path}")
