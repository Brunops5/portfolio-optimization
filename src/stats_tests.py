"""Compare strategy returns with paired Wilcoxon signed-rank tests."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import names


def periodic_returns_from_backtest(
    backtest_df: pd.DataFrame,
    frequency: str,
) -> pd.Series:
    """Compound daily returns into non-overlapping periodic log returns.

    Simple returns are compounded first and converted to log returns second.
    Partial weeks are excluded because cryptocurrency markets trade seven days
    a week, making incomplete weeks unsuitable for paired comparison.
    """
    if names.DAILY_RETURN not in backtest_df:
        raise ValueError("The backtest must contain a daily_return column.")

    daily_returns = backtest_df[names.DAILY_RETURN]
    if daily_returns.empty:
        raise ValueError("Daily returns cannot be empty.")
    if (daily_returns <= -1).any():
        raise ValueError("Daily returns must be greater than -100%.")

    resampler = daily_returns.resample(frequency)
    periodic_simple = (1.0 + daily_returns).resample(frequency).prod() - 1.0
    if frequency.upper().startswith("W-"):
        complete_weeks = resampler.count() == 7
        periodic_simple = periodic_simple.loc[complete_weeks]

    return np.log1p(periodic_simple).dropna()


def wilcoxon_comparison(
    returns_a: pd.Series,
    returns_b: pd.Series,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict:
    """Compare two aligned return series and summarize their differences.

    The test uses paired periods. Fewer than six pairs, or an invalid Wilcoxon
    calculation, produces missing test statistics rather than a false result.
    """
    paired = pd.concat([returns_a, returns_b], axis=1).dropna()
    paired.columns = [label_a, label_b]

    if len(paired) < 6:
        return {
            names.N: len(paired),
            names.WILCOXON_STATISTIC: np.nan,
            names.P_VALUE: np.nan,
            names.MEDIAN_DIFF: np.nan,
            names.MEAN_DIFF: np.nan,
            names.STD_DIFF: np.nan,
            names.MIN_DIFF: np.nan,
            names.MAX_DIFF: np.nan,
            names.WIN_RATE_A: np.nan,
            names.WIN_RATE_B: np.nan,
        }

    differences = paired[label_a] - paired[label_b]
    if np.allclose(differences, 0):
        statistic = 0.0
        p_value = 1.0
    else:
        try:
            statistic, p_value = wilcoxon(
                differences,
                zero_method="wilcox",
                correction=False,
            )
        except ValueError:
            statistic = np.nan
            p_value = np.nan

    wins_a = (differences > 0).sum()
    wins_b = (differences < 0).sum()
    decisive_periods = wins_a + wins_b
    if decisive_periods > 0:
        win_rate_a = wins_a / decisive_periods
        win_rate_b = wins_b / decisive_periods
    else:
        win_rate_a = np.nan
        win_rate_b = np.nan

    return {
        names.N: len(paired),
        names.WILCOXON_STATISTIC: float(statistic),
        names.P_VALUE: float(p_value),
        names.MEDIAN_DIFF: float(differences.median()),
        names.MEAN_DIFF: float(differences.mean()),
        names.STD_DIFF: float(differences.std()),
        names.MIN_DIFF: float(differences.min()),
        names.MAX_DIFF: float(differences.max()),
        names.WIN_RATE_A: float(win_rate_a),
        names.WIN_RATE_B: float(win_rate_b),
    }


def _holm_adjustment(p_values: pd.Series) -> pd.Series:
    """Adjust p-values for multiple comparisons using Holm's method."""
    valid = p_values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    running_max = 0.0
    number_of_tests = len(valid)

    for rank, (index, p_value) in enumerate(valid.items()):
        multiplier = number_of_tests - rank
        adjusted_value = min(1.0, multiplier * p_value)
        running_max = max(running_max, adjusted_value)
        adjusted.loc[index] = running_max

    return adjusted


def compare_all_strategies(
    backtest_results: Dict[str, pd.DataFrame],
    frequency: str = "ME",
) -> pd.DataFrame:
    """Run a paired Wilcoxon comparison for every pair of strategies."""
    print("\n" + "=" * 60)
    print(f"STEP 4 — WILCOXON TEST ({frequency})")
    print("=" * 60)

    periodic = {
        name: periodic_returns_from_backtest(dataframe, frequency)
        for name, dataframe in backtest_results.items()
    }
    strategy_names = list(periodic)
    rows = []

    for first_index in range(len(strategy_names)):
        for second_index in range(first_index + 1, len(strategy_names)):
            strategy_a = strategy_names[first_index]
            strategy_b = strategy_names[second_index]
            stats = wilcoxon_comparison(
                periodic[strategy_a],
                periodic[strategy_b],
                label_a=strategy_a,
                label_b=strategy_b,
            )
            stats[names.STRATEGY_A] = strategy_a
            stats[names.STRATEGY_B] = strategy_b
            stats[names.RETURN_FREQUENCY] = frequency
            rows.append(stats)

            print(f"\n{strategy_a} vs {strategy_b}:")
            print(
                f"  n={stats[names.N]}, "
                f"W={stats[names.WILCOXON_STATISTIC]:.2f}, "
                f"p={stats[names.P_VALUE]:.4f}"
            )
            print(
                f"  Mean difference: {stats[names.MEAN_DIFF] * 100:.2f}%, "
                f"median: {stats[names.MEDIAN_DIFF] * 100:.2f}%"
            )
            print(
                f"  Win rate {strategy_a}: "
                f"{stats[names.WIN_RATE_A] * 100:.1f}% | "
                f"{strategy_b}: {stats[names.WIN_RATE_B] * 100:.1f}%"
            )

    results = pd.DataFrame(rows)
    results[names.P_VALUE_HOLM] = _holm_adjustment(results[names.P_VALUE])
    columns = [
        names.STRATEGY_A,
        names.STRATEGY_B,
        names.RETURN_FREQUENCY,
        names.N,
        names.WILCOXON_STATISTIC,
        names.P_VALUE,
        names.P_VALUE_HOLM,
        names.MEAN_DIFF,
        names.MEDIAN_DIFF,
        names.STD_DIFF,
        names.MIN_DIFF,
        names.MAX_DIFF,
        names.WIN_RATE_A,
        names.WIN_RATE_B,
    ]
    return results[columns]


def save_wilcoxon_results(
    df: pd.DataFrame,
    outdir: Path,
    stem: str = "wilcoxon_results",
) -> None:
    """Save statistical comparisons as CSV and Markdown tables."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    markdown_path = outdir / f"{stem}.md"
    with markdown_path.open("w", encoding="utf-8") as file:
        file.write("# Wilcoxon Signed-Rank Test Results\n\n")
        file.write(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"[SAVED] {markdown_path}")
